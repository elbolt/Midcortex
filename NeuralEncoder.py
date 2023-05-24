import os
import numpy as np
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.model_selection import KFold
from tqdm import tqdm

mne.set_log_level('ERROR')


class NeuralEncoder:
    def __init__(self, subject_id, method, alphas=None):
        self.subject_id = subject_id
        self.method = method
        self.file_path = '/Volumes/NeuroSSD/Midcortex/'
        self.alphas = alphas

        if method == 'cortical':
            self.filename = os.path.join(self.file_path, 'cortex_encoder', f'{subject_id}.npy')
            self.fs = 128
            self.trf_min, self.trf_max = -200e-3, 500e-3

            self.speech = np.load('audio/low_envelopes.npy')
            self.eeg = np.load(self.filename)

        elif method == 'subcortical':
            self.filename = os.path.join(self.file_path, 'subcortex_encoder', f'{subject_id}.npy')
            self.fs = 4096
            self.trf_min, self.trf_max = -5e-3, 20e-3

            self.speech = np.load('audio/rectified_audios.npy')
            self.eeg = np.load(self.filename)[..., 29:31]  # remove later

        else:
            raise ValueError(f'`{method}` invalid, method must be `cortical` or `subcortical`.')

        # Determine kernel size (based on ReceptiveField source code)
        lags_future = int(np.round(self.trf_min * self.fs))
        lags_past = int((np.round(self.trf_max * self.fs) + 1))
        self.kernel_size = abs(lags_future) + lags_past

        # Subject-specific problems
        if self.subject_id == 'pilot03':
            # For this participant, we forgot to record right away, this is why the first trial is missing.
            self.speech = np.delete(self.speech, 0, axis=1)

    def fit(self):
        speech = self.speech
        eeg = self.eeg

        n_splits = 10
        n_splits_alpha = 5

        # Initialize arrays to store scores and kernel weights
        kernels = np.full((n_splits, eeg.shape[2], self.kernel_size), np.nan)
        scores = np.full((n_splits, eeg.shape[2]), np.nan)

        # Initialize outer cross-validation
        k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
        best_alphas = np.full((n_splits), np.nan)

        for fold_idx, (train_indices, test_indices) in tqdm(
                enumerate(k_folds.split(np.arange(eeg.shape[1]))), total=n_splits):

            # Split data into training/test – ReceptiveField shape is (n_times, n_epochs, n_channels)
            speech_train = speech[:, train_indices, :]
            speech_test = speech[:, test_indices, :]
            eeg_train = eeg[:, train_indices, :]
            eeg_test = eeg[:, test_indices, :]

            # Initialize the inner loop for hyperparameter tuning
            k_folds_alpha = KFold(n_splits=n_splits_alpha, shuffle=True, random_state=2023)

            all_scores = np.full((len(self.alphas)), np.nan)

            # Iterate over alphas for regularization
            for alpha_idx, alpha in enumerate(self.alphas):

                mean_inner_scores = np.full((n_splits_alpha), np.nan)

                # Iterate over inner folds for cross-validation
                for alpha_fold_idx, (train_indices_alpha, validation_indices_alpha) in enumerate(
                        k_folds_alpha.split(np.arange(speech_train.shape[1]))):

                    # Initialize alpha tuning model
                    alpha_estimator = TimeDelayingRidge(
                        tmin=self.trf_min,
                        tmax=self.trf_max,
                        sfreq=self.fs,
                        reg_type='laplacian',
                        alpha=alpha
                    )

                    alpha_model = ReceptiveField(
                        tmin=self.trf_min,
                        tmax=self.trf_max,
                        estimator=alpha_estimator,
                        sfreq=self.fs,
                        scoring='corrcoef'
                    )

                    # Split training data into training/validation
                    alpha_speech_train = speech_train[:, train_indices_alpha, :]
                    alpha_speech_validation = speech_train[:, validation_indices_alpha, :]
                    alpha_eeg_train = eeg_train[:, train_indices_alpha, :]
                    alpha_eeg_validation = eeg_train[:, validation_indices_alpha, :]

                    alpha_model.fit(alpha_speech_train, alpha_eeg_train)

                    # Store score of current fold
                    inner_score = alpha_model.score(alpha_speech_validation, alpha_eeg_validation)  # (32,)
                    mean_inner_score = inner_score.mean()  # (1,)
                    mean_inner_scores[alpha_fold_idx] = mean_inner_score

                mean_alpha_score = mean_inner_scores.mean()
                all_scores[alpha_idx] = mean_alpha_score

            # Find the best alpha
            best_alpha_idx = np.where(all_scores == np.max(all_scores))[0][0]
            best_alpha = self.alphas[best_alpha_idx]
            best_alphas[fold_idx] = best_alpha

            # Use the best alpha found in the inner loop in the outer loop and update lag range!

            # Initialize model
            estimator = TimeDelayingRidge(
                tmin=self.trf_min,
                tmax=self.trf_max,
                sfreq=self.fs,
                reg_type='laplacian',
                alpha=best_alpha
            )
            model = ReceptiveField(
                tmin=self.trf_min,
                tmax=self.trf_max,
                estimator=estimator,
                sfreq=self.fs,
                scoring='corrcoef'
            )

            model.fit(speech_train, eeg_train)

            # Evaluate the model on the outer test set
            score = model.score(speech_test, eeg_test)
            kernel = model.coef_.squeeze()

            # Cache model scores and model weights
            scores[fold_idx, :] = score
            kernels[fold_idx, ...] = kernel

        # Calculate the average scores and kernel weights
        mean_scores = np.nanmean(scores, axis=0)
        mean_kernels = np.nanmean(kernels, axis=0)

        return mean_kernels, mean_scores, best_alphas

    def predict(self, X):
        # Predict the EEG response to speech input
        return self.model.predict(X)
