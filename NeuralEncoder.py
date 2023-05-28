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
            self.trf_min, self.trf_max = -5e-3, 15e-3

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
        alpha_history = []
        outer_alpha_history = []

        # Initialize outer cross-validation
        k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=2023)

        for fold_idx, (train_indices, test_indices) in tqdm(
                enumerate(k_folds.split(np.arange(eeg.shape[1]))), total=n_splits):

            # Split data into training/test – ReceptiveField shape is (n_times, n_epochs, n_channels)
            speech_train = speech[:, train_indices, :]
            speech_test = speech[:, test_indices, :]
            eeg_train = eeg[:, train_indices, :]
            eeg_test = eeg[:, test_indices, :]

            # Compute the mean and standard deviation using the training data of the outer fold, ignoring NaN values
            mean = np.nanmean(speech_train)
            std = np.nanstd(speech_train)

            best_score = -np.inf
            best_alpha = None

            # Iterate over alphas for regularization
            for alpha in self.alphas:

                # Initialize the inner loop for hyperparameter tuning
                k_folds_alpha = KFold(n_splits=n_splits_alpha, shuffle=True, random_state=2023)

                # Iterate over inner folds for cross-validation
                for (train_indices_alpha, validation_indices_alpha) in k_folds_alpha.split(np.arange(speech_train.shape[1])):

                    # Split training data into training/validation
                    alpha_speech_train = speech_train[:, train_indices_alpha, :]
                    alpha_speech_validation = speech_train[:, validation_indices_alpha, :]
                    alpha_eeg_train = eeg_train[:, train_indices_alpha, :]
                    alpha_eeg_validation = eeg_train[:, validation_indices_alpha, :]

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

                    alpha_model.fit(alpha_speech_train, alpha_eeg_train)

                    # Store score of current fold
                    inner_score = alpha_model.score(alpha_speech_validation, alpha_eeg_validation)  # (32,)
                    mean_inner_score = inner_score.mean()  # (1,)

                    if mean_inner_score > best_score:
                        best_score = mean_inner_score
                        best_alpha = alpha

                # Track the best params and their corresponding scores
                alpha_history.append({'alpha': best_alpha, 'score': best_score})

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

            outer_alpha_history.append({'best alpha': best_alpha, 'score': score.mean()})

            # Cache model scores and model weights
            scores[fold_idx, :] = score
            kernels[fold_idx, ...] = kernel

        # Calculate the average scores and kernel weights
        # mean_scores = np.nanmean(scores, axis=0)
        # mean_kernels = np.nanmean(kernels, axis=0)

        self.model = model

        return kernels, scores, alpha_history, outer_alpha_history

    def predict(self, X):
        # Predict the EEG response to speech input
        return self.model.predict(X)


def manual_transform(data, mean, std):
    """
    Manually transform the data using pre-computed mean and standard deviation,
    handling NaN values.
    """
    normalized_data = np.where(np.isnan(data), np.nan, (data - mean) / std)
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    return normalized_data


def manual_fit_transform(data):
    """
    Manually compute the mean and standard deviation of the data, and transform it,
    handling NaN values.
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return manual_transform(data, mean, std)

