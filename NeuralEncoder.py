import os
import numpy as np
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from tqdm import tqdm
from collections import defaultdict
from helpers import electrode_indices

mne.set_log_level('ERROR')


class NeuralEncoder:
    def __init__(self, subject_id, method, alphas=None, shuffle=False):
        self.subject_id = subject_id
        self.subject_seed = 100 + int(subject_id[-2:])
        self.method = method
        self.shuffle = shuffle
        self.file_path = '/Volumes/NeuroSSD/Midcortex/'
        self.alphas = alphas

        # Model parameters
        self.best_alpha = None
        self.history = None
        self.model = None
        self.model_score = None
        self.weights = None
        self.lags = None

        # Set participant-specific random seed
        np.random.seed(self.subject_seed)

        if shuffle not in [True, False]:
            raise ValueError(f'`{self.shuffle}` invalid, method must be `True` or `False`.')

        if method == 'cortical':
            self.filename = os.path.join(self.file_path, 'cortex_encoder', f'{subject_id}.npy')
            self.fs = 128
            self.trf_min, self.trf_max = -100e-3, 400e-3

            self.speech = np.load('audio/low_envelopes.npy')
            self.eeg = np.load(self.filename)

        elif method == 'subcortical':
            self.filename = os.path.join(self.file_path, 'subcortex_encoder', f'{subject_id}.npy')
            self.fs = 4096
            self.trf_min, self.trf_max = -3e-3, 12e-3

            self.speech = np.load('audio/rectified_audios.npy')
            self.eeg = np.load(self.filename)[..., electrode_indices]

        else:
            raise ValueError(f'`{method}` invalid, method must be `cortical` or `subcortical`.')

        if shuffle is True:
            # Shuffle the 3D array `self.speech` along the second axis (i.e. the trials)
            idx = np.random.permutation(self.speech.shape[1])
            self.speech = self.speech[:, idx, :]
            print(f'Shuffled speech data for {self.method} encoder.')
        else:
            pass

        # Subject-specific problems
        if self.subject_id == 'pilot03':
            # For this participant, we forgot to record right away, this is why the first trial is missing.
            self.speech = np.delete(self.speech, 0, axis=1)

        if self.subject_id in ['pilot02', 'pilot03']:
            self.subject_seed = 300 + int(subject_id[-2:])

    def fit(self):
        # Split the trial indices into train and test indices
        trial_indices = np.arange(self.eeg.shape[1])
        modified_trial_indices = trial_indices[1:-1]  # exclude first & last index from split

        train_indices, test_indices = train_test_split(
            modified_trial_indices,
            test_size=0.20,  # 80/20 split
            random_state=self.subject_seed
        )

        train_indices = np.concatenate([trial_indices[:1], train_indices, trial_indices[-1:]])
        np.random.shuffle(train_indices)  # shuffle again

        # Split data into training/test – ReceptiveField shape is (n_times, n_epochs, n_channels)
        speech_train = self.speech[:, train_indices, :]
        speech_test = self.speech[:, test_indices, :]
        eeg_train = self.eeg[:, train_indices, :]
        eeg_test = self.eeg[:, test_indices, :]

        # Compute the mean and standard deviation using the training data of the outer fold, ignoring NaN values
        mean = np.nanmean(speech_train)
        std = np.nanstd(speech_train)

        # Normalize the training data and replace NaN with 0s using manual_transform()
        speech_train_normalized = manual_transform(speech_train, mean, std)

        best_score = -np.inf
        best_alpha = None
        loo = LeaveOneOut()
        history = []

        # Create a dictionary to keep track of the cumulative score for each alpha
        alpha_scores = defaultdict(float)

        # Iterate over alphas for regularization
        for alpha in self.alphas:
            train_set_indices = np.arange(eeg_train.shape[1])

            # Keep track of the number of folds for this alpha
            fold_count = 0

            for _, (train_indices_loo, val_index) in tqdm(enumerate(loo.split(train_set_indices)), total=len(train_set_indices)):
                speech_train_loo = speech_train_normalized[:, train_indices_loo, :]
                speech_val = speech_train_normalized[:, val_index, :]
                eeg_train_loo = eeg_train[:, train_indices_loo, :]
                eeg_val = eeg_train[:, val_index, :]

                # Initialize alpha tuning model
                estimator = TimeDelayingRidge(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    sfreq=self.fs,
                    reg_type='laplacian',
                    alpha=alpha
                )

                model = ReceptiveField(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    estimator=estimator,
                    sfreq=self.fs,
                    scoring='corrcoef'
                )

                model.fit(speech_train_loo, eeg_train_loo)

                # Store score of current fold
                inner_score = model.score(speech_val, eeg_val)
                mean_inner_score = inner_score.mean()

                # Add the score to the cumulative total for this alpha
                alpha_scores[alpha] += mean_inner_score
                fold_count += 1

            # After all folds for this alpha, compute the average score
            avg_score = alpha_scores[alpha] / fold_count

            history.append({'alpha': alpha, 'average score': avg_score})

            # If this alpha's average score is better than the best seen so far, update best_score and best_alpha
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

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

        self.best_alpha = best_alpha
        self.history = history

        # Evaluate the model on the outer test set and cache
        self.model = model
        self.model_score = model.score(speech_test_normalized, eeg_test)
        self.weights = model.coef_.squeeze().T
        self.lags = model.delays_ / model.sfreq * 1000


def manual_transform(data, mean, std):
    """
    Manually transform the data using pre-computed mean and standard deviation,
    handling NaN values.
    """
    normalized_data = (data - mean) / std
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)

    return normalized_data
