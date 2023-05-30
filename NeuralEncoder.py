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
            self.trf_min, self.trf_max = -100e-3, 400e-3
            self.xval_min, self.xval_max = self.trf_min, self.trf_max

            self.speech = np.load('audio/low_envelopes.npy')
            self.eeg = np.load(self.filename)  # (n_times, n_epochs, n_channels)
            # # Shuffle the 3D array `self.speech` along the second axis (i.e. the trials)
            # idx = np.random.permutation(self.speech.shape[1])
            # self.speech = self.speech[:, idx, :]

        elif method == 'subcortical':
            self.filename = os.path.join(self.file_path, 'subcortex_encoder', f'{subject_id}.npy')
            self.fs = 4096
            self.trf_min, self.trf_max = -5e-3, 15e-3

            self.speech = np.load('audio/rectified_audios.npy')
            self.eeg = np.load(self.filename)[..., 29:31]  # remove later!

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

        n_splits = 10  # self.eeg.shape[1]
        n_splits_alpha = 5  # speech_train.shape[1]

        # Initialize arrays to store scores and kernel weights
        kernels = np.full((n_splits, eeg.shape[2], self.kernel_size), np.nan)
        scores = np.full((n_splits, eeg.shape[2]), np.nan)
        history = []

        # Initialize outer cross-validation
        k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        trial_indices = np.arange(eeg.shape[1])

        for fold_idx, (train_indices, test_indices) in tqdm(enumerate(k_folds.split(trial_indices)), total=n_splits):

            # Split data into training/test – ReceptiveField shape is (n_times, n_epochs, n_channels)
            speech_train = speech[:, train_indices, :]
            speech_test = speech[:, test_indices, :]
            eeg_train = eeg[:, train_indices, :]
            eeg_test = eeg[:, test_indices, :]

            # Compute the mean and standard deviation using the training data of the outer fold, ignoring NaN values
            mean = np.nanmean(speech_train)
            std = np.nanstd(speech_train)

            # Normalize the training data and replace NaN with 0s using manual_transform()
            speech_train_normalized = manual_transform(speech_train, mean, std)

            best_score = -np.inf
            best_alpha = None

            # Iterate over alphas for regularization
            for alpha in self.alphas:

                # Initialize the inner loop for hyperparameter tuning
                k_folds_alpha = KFold(n_splits=n_splits_alpha, shuffle=True, random_state=42)

                alpha_trials_indeces = np.arange(speech_train.shape[1])

                # Iterate over inner folds for cross-validation
                for inner_i, (train_indices_alpha, validation_indices_alpha) in enumerate(k_folds_alpha.split(alpha_trials_indeces)):

                    # Split training data into training/validation
                    alpha_speech_train = speech_train_normalized[:, train_indices_alpha, :]
                    alpha_speech_validation = speech_train_normalized[:, validation_indices_alpha, :]
                    alpha_eeg_train = eeg_train[:, train_indices_alpha, :]
                    alpha_eeg_validation = eeg_train[:, validation_indices_alpha, :]

                    # Initialize alpha tuning model
                    alpha_estimator = TimeDelayingRidge(
                        tmin=self.xval_min,
                        tmax=self.xval_max,
                        sfreq=self.fs,
                        reg_type='laplacian',
                        alpha=alpha
                    )

                    alpha_model = ReceptiveField(
                        tmin=self.xval_min,
                        tmax=self.xval_max,
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

                # # Track the best params and their corresponding scores
                # history.append({'alpha': best_alpha, 'score': best_score})
                self.alpha_model = alpha_model

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

            model.fit(speech_train_normalized, eeg_train)

            # Normalize the testing data using manual_transform()
            speech_test_normalized = manual_transform(speech_test, mean, std)

            # Evaluate the model on the outer test set
            score = model.score(speech_test_normalized, eeg_test)
            kernel = model.coef_.squeeze()

            history.append({'best alpha': best_alpha, 'score': score.mean()})

            # Cache model scores and model weights
            scores[fold_idx, :] = score
            kernels[fold_idx, ...] = kernel

            del best_alpha, best_score

        # Calculate the average scores and kernel weights
        # mean_scores = np.nanmean(scores, axis=0)
        # mean_kernels = np.nanmean(kernels, axis=0)

        self.model = model

        return kernels, scores, history


def manual_transform(data, mean, std):
    """
    Manually transform the data using pre-computed mean and standard deviation,
    handling NaN values.
    """
    normalized_data = (data - mean) / std
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)

    return normalized_data
