import numpy as np
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from sklearn.model_selection import KFold
from tqdm import tqdm
from helpers import auditory_cluster, electrodes

mne.set_log_level('ERROR')


class NeuralEncoder:
    def __init__(self, subject_id, method, regularization=True, alphas=None, init_estimator=0.0):
        self.subject_id = subject_id
        self.method = method
        self.regularization = regularization
        if regularization not in [True, False]:
            raise ValueError('Regularization must be `True` or `False``.')
        self.init_estimator = init_estimator

        self.alphas = alphas

        if method == 'cortical':
            self.fs = 128
            self.trf_min, self.trf_max = -0.300, 0.800
            self.xval_min, self.xval_max = -0.100, 0.300
            self.speech = np.load('audio/low_envelopes.npy')
            self.cluster = auditory_cluster
            indices = [electrodes.index(electrode) for electrode in self.cluster]
            self.eeg = np.load(f'eeg/cortex/{subject_id}.npy')[..., indices]

        elif method == 'subcortical':
            self.fs = 4096
            self.trf_min, self.trf_max = -0.006, 0.026
            self.xval_min, self.xval_max = 0.002, 0.014
            self.speech = np.load('audio/rectified_audios.npy')
            self.cluster = ['Cz']  # ['Pz', 'Fz', 'Cz']
            self.eeg = np.load(f'eeg/subcortex/{subject_id}.npy')  # [..., 2][..., np.newaxis]  # Cz only

        else:
            raise ValueError(f'`{method}` invalid, method must be `cortical` or `subcortical`.')

        if self.subject_id == 'pilot03':
            # For this participant, we forgot to record right away, this is why the first trial is missing.
            self.speech = np.delete(self.speech, 0, axis=1)

        self.kernel_size = int(np.ceil((self.trf_max - self.trf_min) * self.fs))

        self.estimator = TimeDelayingRidge(
            tmin=self.trf_min,
            tmax=self.trf_max,
            sfreq=self.fs,
            reg_type='laplacian',
            alpha=0.0
        )

        self.model = ReceptiveField(
            tmin=None,
            tmax=None,
            estimator=None,
            sfreq=self.fs,
            scoring='corrcoef'
        )

        self.mean_scores_alpha = None
        self.best_alpha = None

    def fit(self):
        X = self.speech
        y = self.eeg

        if self.regularization:
            n_splits_outer = 10
            n_splits_inner = 5
        elif not self.regularization:
            n_splits_outer = self.speech.shape[1]

        # Initialize arrays to store scores and kernel weights
        kernels = np.full((n_splits_outer, y.shape[2], self.kernel_size), np.nan)
        scores = np.full((n_splits_outer, y.shape[2]), np.nan)

        # Initialize outer cross-validation
        outer_k_folds = KFold(n_splits=n_splits_outer, shuffle=True, random_state=2004)

        for i, (train_indices, test_indices) in tqdm(
                enumerate(outer_k_folds.split(np.arange(y.shape[1]))), total=n_splits_outer
        ):
            # ReceptiveField shape is (n_times, n_epochs, n_channels)
            speech_train_outer = X[:, train_indices, :]
            speech_test = X[:, test_indices, :]
            eeg_train_outer = y[:, train_indices, :]
            eeg_test = y[:, test_indices, :]

            # Initialize the inner loop for hyperparameter tuning
            best_alpha = None
            best_inner_score = float('-inf')
            mean_scores_alpha = np.full((len(self.alphas)), np.nan)

            if self.regularization:
                inner_k_folds = KFold(n_splits=n_splits_inner, shuffle=True, random_state=2004)

                # Iterate over alphas for regularization
                for alpha_idx, alpha in enumerate(self.alphas):
                    inner_scores = []

                    self.estimator.set_params(
                        tmin=self.xval_min,
                        tmax=self.xval_max,
                        alpha=alpha
                    )
                    self.model.set_params(
                        tmin=self.xval_min,
                        tmax=self.xval_max,
                        estimator=self.estimator
                    )

                    # Iterate over inner folds for cross-validation
                    for _, (inner_train_indices, inner_val_indices) in enumerate(
                            inner_k_folds.split(np.arange(speech_train_outer.shape[1]))):

                        # (n_times, n_epochs, n_channels)!
                        speech_train = speech_train_outer[:, inner_train_indices, :]
                        speech_val = speech_train_outer[:, inner_val_indices, :]
                        eeg_train = eeg_train_outer[:, inner_train_indices, :]
                        eeg_val = eeg_train_outer[:, inner_val_indices, :]

                        self.model.fit(speech_train, eeg_train)
                        inner_score = self.model.score(speech_val, eeg_val)  # score array for each electrode
                        inner_scores.append(inner_score)  # lists with 5 score arrays

                    mean_inner_score = np.mean(inner_scores)  # mean score of electrodes for each of the 5 splits

                    # Cache mean scroes across alphas for plotting
                    mean_scores_alpha[alpha_idx] = mean_inner_score

                    # Update `best_alpha` if the score gets bigger.
                    if mean_inner_score > best_inner_score:
                        best_inner_score = mean_inner_score
                        best_alpha = alpha

                # Use the best alpha found in the inner loop in the outer loop and update lag range
                self.estimator.set_params(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    alpha=best_alpha
                )
                self.model.set_params(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    estimator=self.estimator
                )

            # If regularization is not used, directly set estimator and model paramets to full lags range
            elif not self.regularization:
                self.estimator.set_params(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    alpha=self.init_estimator
                )
                self.model.set_params(
                    tmin=self.trf_min,
                    tmax=self.trf_max,
                    estimator=self.estimator
                )

            self.model.fit(speech_train_outer, eeg_train_outer)

            # Evaluate the model on the outer test set
            score = self.model.score(speech_test, eeg_test)
            kernel = self.model.coef_.squeeze()

            # Cache model scores and model weights
            scores[i, :] = score
            kernels[i, ...] = kernel

        # Calculate the average scores and kernel weights
        mean_scores = np.nanmean(scores, axis=0)
        mean_kernels = np.nanmean(kernels, axis=0)

        self.mean_scores_alpha = mean_scores_alpha

        if self.regularization:
            self.best_alpha = best_alpha
        elif not self.regularization:
            self.best_alpha = self.init_estimator

        return mean_kernels, mean_scores

    def predict(self, X):
        # Predict the EEG response to speech input
        return self.model.predict(X)
