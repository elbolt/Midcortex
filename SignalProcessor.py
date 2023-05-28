import os
import numpy as np
import textgrid

from scipy import signal
from scipy.io import wavfile


class SoundProcessor():
    def __init__(self, file, method):
        self.method = method
        self.file_path = f'audio/raw/{file}.wav'
        self.textgrid = textgrid.TextGrid.fromFile(f'audio/raw/{file}.TextGrid')
        self.fs, self.data = wavfile.read(self.file_path)

        if self.method == 'cortical':
            self.fs_env = 12000
            self.fs_goal = 128
            self.envelope = None
            self.band_passed_signal = None

        elif self.method == 'subcortical':
            self.fs_goal = 4096
            self.high_passed_signal = None

        else:
            raise ValueError(f'`{self.method}` must be `cortical` or `subcortical`.')

    def remove_silent_onset(self):
        """
        Removes the silent onset from the beginning of the audio file.
        """
        silent_seconds = self.textgrid[0][0].time
        silent_samples = int(np.round(silent_seconds * self.fs))
        self.data = self.data[silent_samples:]

    def extract_gammatone_envelope(self):
        """
        Gammatone envelope exctraction procedure:
        1. Downsample signal to 12 000 Hz (anti-aliasing filter at 6000 Hz: two-pass zero-phase FIR
        controlled for filter delay, order 98.
        2. Pass through Gammatone filterbank with 24 filter bands from 100--4000 Hz.
        3. Full-wave rectify, compress by 0.2, and average the filtered signals across center frequencies.
        """
        if self.method != 'cortical':
            raise MissingAttributeError('Method must be set to `cortical` to call this method')

        fs = self.fs
        data = self.data
        fs_env = self.fs_env

        nyq_rate = fs / 2
        filter_freq = 6000
        cutoff_freq = filter_freq / nyq_rate
        order = 98
        fir_coeff = signal.firwin(order, cutoff_freq)

        data = signal.filtfilt(fir_coeff, 1.0, data)
        data = signal.resample(data, int(len(data) / fs * fs_env))
        # self.filter1 = data

        num_filters = 24
        freq_range = (100, 4000)
        filterbank = SoundProcessor.gammatone_filterbank(
            sampling_rate=12000,
            num_filters=num_filters,
            freq_range=freq_range
        )
        envelope = np.vstack([signal.lfilter(filterbank[i, :], 1.0, data) for i in range(num_filters)])

        compression = 0.3
        envelope = np.abs(envelope)
        envelope = np.power(envelope, compression)
        envelope = np.mean(envelope, axis=0)

        self.envelope = envelope

    def band_pass_signal(self):
        """
        Band-pass filter procedure:
        """
        if self.method != 'cortical':
            raise MissingAttributeError('Method must be set to `cortical` to call this method')

        fs_goal = self.fs_goal
        fs_env = self.fs_env
        data = self.envelope

        """
        1. Downsample signal to 128 Hz (anti-aliasing filter at 51.2 Hz: one-pass zero-phase hamming-windowed sinc FIR
        controlled for filter delay, order 3067 (Florine: 3094), transition width 12.8 Hz).
        """
        nyq_rate = fs_env / 2
        cutoff_freq = 51.2
        trans_width = 12.8
        order = determine_filter_order(fs_env, trans_width)

        freq_points = [0, cutoff_freq, cutoff_freq + trans_width, nyq_rate]
        gain_points = [1, 1, 0, 0]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            fs=fs_env,
            window='hamming'
        )

        data = signal.filtfilt(fir_coeff, 1.0, data)
        data = signal.resample(data, int(len(data) / fs_env * fs_goal), window=('kaiser', 5.0))
        # self.filter2 = data  # control plots

        del fs_env, cutoff_freq, trans_width, nyq_rate, order, freq_points, gain_points, fir_coeff

        """
        2. High-pass filter the signal at 0.5 Hz (one-pass zero-phase hamming-windowed sinc FIR controlled for filter
        delay, order 417 (Florine: 424), transition width 1 Hz).
        """
        nyq_rate = fs_goal / 2
        filter_freq = 0.5
        trans_width = 1.0
        order = determine_filter_order(fs_goal, trans_width)

        highpass_taps = signal.firwin(
            order,
            filter_freq,
            fs=fs_goal,
            pass_zero=False,
            window='hamming'
        )

        data = signal.filtfilt(highpass_taps, 1.0, data)
        # self.filter3 = data  # control plots

        del filter_freq, trans_width, order, highpass_taps

        """
        3. Band-pass filter signal between 1–9 Hz.
        - high-pass filter: one-pass zero- phase, order 209 (212), transition width 2 Hz
        - low-pass filter: one-pass zero-phase, order 189 (188), transition width 2.2 Hz).
        """
        lowcut = 1.0
        trans_width_high = 2.0
        order_high = determine_filter_order(fs_goal, trans_width_high)

        highpass_taps = signal.firwin(
            order_high,
            lowcut,
            fs=fs_goal,
            pass_zero=False,
            window='hamming'
        )

        highcut = 9.0
        trans_width_low = 2.2
        order_low = determine_filter_order(fs_goal, trans_width_low)

        lowpass_taps = signal.firwin(
            order_low,
            highcut,
            fs=fs_goal,
            pass_zero=True,
            window='hamming'
        )

        data = signal.filtfilt(highpass_taps, 1.0, data)  # high-pass filter
        data = signal.filtfilt(lowpass_taps, 1.0, data)  # low-pass filter

        self.band_passed_signal = data

    def high_pass_signal(self):
        """
        High-pass filter procedure.
        """
        if self.method != 'subcortical':
            raise MissingAttributeError('Method must be set to `subcortical` to call this method')

        fs = self.fs
        fs_goal = self.fs_goal
        data = self.data

        """
        Down-sample signal from 44100 to 4096 Hz (anti-aliasing filter at 1638: one-pass zero-phase hamming-windowed
        sinc FIR controlled for filter delay, order 351 (Florine: 356), transition width 409.6 Hz).
        """
        cutoff_freq = 1638.0
        trans_width = 409.6
        nyq_rate = fs / 2
        order = determine_filter_order(fs, trans_width)

        freq_points = [0, cutoff_freq, cutoff_freq + trans_width, nyq_rate]
        gain_points = [1, 1, 0, 0]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            window='hamming',
            fs=fs
        )

        data = signal.filtfilt(fir_coeff, 1.0, data)
        data = signal.resample(data, int(len(data) * fs_goal / fs), window=('kaiser', 5.0))
        # self.filter2 = data  # control plots

        del fs, cutoff_freq, trans_width, nyq_rate, order, freq_points, gain_points, fir_coeff

        """
        High-pass filter at 80 Hz (one-pass zero-phase Hamming-windowed sinc FIR corrected for filter delay, order
        671 (Florine: 676), transition width 20 Hz.
        """
        cutoff_freq = 80.0
        trans_width = 20.0
        nyq_rate = fs_goal / 2
        order = determine_filter_order(fs_goal, trans_width)

        freq_points = [0, cutoff_freq - (trans_width / 2), cutoff_freq + (trans_width / 2), nyq_rate]
        gain_points = [0, 0, 1, 1]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            window='hamming',
            fs=fs_goal
        )

        data = signal.filtfilt(fir_coeff, 1.0, data)
        # self.filter3 = data  # control plots

        """
        Half-wave rectification.
        """
        data = np.maximum(data, 0)

        self.high_passed_signal = data

    def cut_ends(self, cut=1):
        """
        Cuts the ends of the specified sound file.

        Parameters
        ----------
        cut : float, optional
            The amount of time to cut from each end of the sound file (in seconds). Default is 1.
        """
        if self.method == 'cortical':
            fs = self.fs_goal
            data = self.band_passed_signal
        elif self.method == 'subcortical':
            fs = self.fs_goal
            data = self.high_passed_signal

        # Cut onset and cut end
        samples_to_cut = cut * fs
        data = data[samples_to_cut:(len(data) - samples_to_cut)]

        if self.method == 'cortical':
            self.band_passed_signal = data
        elif self.method == 'subcortical':
            self.high_passed_signal = data

    @staticmethod
    def gammatone_filterbank(sampling_rate, num_filters, freq_range):
        """
        Generate a Gammatone filterbank (Glasberg & Moore, 1990).

        This function generates a Gammatone filterbank, which is a set of bandpass filters that simulate the frequency
        response of the human auditory system. The filters are designed to be similar to the response of the cochlea,
        which is the organ in the inner ear responsible for processing sound.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate of signal to be filtered.
        num_filters : int
            The number of filters in the filterbank.
        freq_range : tuple of (min_freq, max_freq)
            Frequency range of the filter.

        Returns
        -------
        tuple of (filter_bank, center_freqs)
            A tuple of (filter_bank, center_freqs), where filter_bank is a matrix of shape (num_filters, n), and
            center_freqs is a vector of shape (num_filters,).

        References
        ----------
        Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory filter shapes from notched-noise data.
        Hearing Research, 47(1-2), 103-138. doi:10.1016/0378-5955(90)90170-T
        """

        # Compute ERB (Equivalent Rectangular Bandwidth)
        min_freq, max_freq = freq_range
        erb_min = 24.7 * (4.37 * min_freq / 1000 + 1)
        erb_max = 24.7 * (4.37 * max_freq / 1000 + 1)

        # Compute center frequencies in ERB and Hz
        center_freqs_erb = np.linspace(erb_min, erb_max, num_filters)
        center_freqs_hz = (center_freqs_erb / 24.7 - 1) / 4.37 * 1000

        # Compute filter bandwidths and Q factors
        q = 1.0 / (center_freqs_erb * 0.00437 + 1.0)
        bandwidths = center_freqs_hz * q

        # Compute filter bank
        filter_bank = np.zeros((num_filters, 4))
        t = np.arange(4) / sampling_rate
        for i in range(num_filters):
            c = 2 * np.pi * center_freqs_hz[i]
            b = 1.019 * 2 * np.pi * bandwidths[i]

            # Compute envelope and sine wave
            envelope = (c ** 4) / (b * np.math.factorial(4)) * t ** 3 * np.exp(-b * t)
            sine_wave = np.sin(c * t)

            # Apply envelope to sine wave and store in filter bank
            filter_bank[i, :] = sine_wave * envelope

        return filter_bank


class NeuralProcessor():
    def __init__(self, subject, method):
        self.fs = 4096
        self.method = method

        folder = '/Volumes/NeuroSSD/Midcortex/cleaned_arrays'
        self.filename = os.path.join(folder, f'{subject}.npy')

        if method == 'cortical':
            self.fs_goal = 128
            self.raw_data = np.load(self.filename)
            self.data = None

        elif method == 'subcortical':
            self.raw_data = np.load(self.filename)
            self.data = None

        else:
            raise ValueError(f'`{method}` must be `cortical` or `subcortical`.')

    def resample(self):
        """
        Low-pass filter procedure:
        """
        if self.method != 'cortical':
            raise MissingAttributeError('Method must be set to `cortical` to call this method')

        fs_goal = self.fs_goal
        fs = self.fs
        data = self.raw_data

        """
        1. Downsample signal to 128 Hz (anti-aliasing filter at 51.2 Hz: one-pass zero-phase hamming-windowed sinc FIR
        controlled for filter delay, order 4391 (Florine: 4424), transition width 12.8 Hz).

        A common method to estimate the filter order is based on the empirical relationship derived by Kaiser.
        order = (A - 8) * (fs / (22 * trans_width)).
        """
        nyq_rate = fs / 2
        filter_freq = 51.2
        trans_width = 12.8
        order = determine_filter_order(fs, trans_width, attenuation=310)  # higher attenuation

        freq_points = [0, filter_freq, filter_freq + trans_width, nyq_rate]
        gain_points = [1, 1, 0, 0]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            fs=fs,
            window='hamming'
        )

        data = signal.filtfilt(fir_coeff, 1.0, data, axis=2)

        new_num_samples = int(data.shape[2] / fs * fs_goal)
        resampled_data = np.zeros((data.shape[0], data.shape[1], new_num_samples))

        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                resampled_data[trial, channel] = signal.resample(
                    data[trial, channel],
                    new_num_samples,
                    window=('kaiser', 5.0)
                )

        data = resampled_data

        del nyq_rate, fs, filter_freq, trans_width, order, fir_coeff

        """
        2. High-pass filter the signal at 0.5 Hz (one-pass zero-phase hamming-windowed sinc FIR controlled for filter
        delay, order 417 (Florine: 424), transition width 1 Hz).
        """
        filter_freq = 0.5
        trans_width = 1.0
        order = determine_filter_order(fs_goal, trans_width)

        highpass_taps = signal.firwin(
            order,
            filter_freq,
            fs=fs_goal,
            pass_zero=False,
            window='hamming'
        )

        data = signal.filtfilt(highpass_taps, 1.0, data, axis=2)

        self.data = data

        del filter_freq, trans_width, order, highpass_taps

    def clean_signal(self, window_duration=1.0):
        """
        Function to clean signal by removing artifacts.

        Parameters:
        signal (np.ndarray): The input signal. Should be a 3D numpy array of shape (n_signals, n_channels, n_samples)
        fs (int): The sampling rate of the signal
        window_duration (float): The duration of the window for artifact removal, in seconds

        Returns:
        np.ndarray: The cleaned signal
        """
        if self.method == 'cortical':
            fs = self.fs_goal
        elif self.method == 'subcortical':
            fs = self.fs

        data = self.data

        # Compute the window size in samples
        window_samples = int(window_duration * fs)

        # Compute padding size and apply padding
        padding_samples = window_samples // 2
        padding = np.tile(np.mean(data, axis=2, keepdims=True), (1, 1, padding_samples))
        data_padded = np.concatenate([padding, data, padding], axis=2)

        # Copy the padded signal for processing
        data_clean = np.copy(data_padded)

        # Compute mean and standard deviation for all channels and signals at once
        mean = np.mean(data_padded, axis=2)
        std = np.std(data_padded, axis=2)

        # Identify the indices of the signal that are > 5 std from the mean for all channels and signals
        indices_list = np.abs(data_padded - mean[:, :, None]) > 5 * std[:, :, None]

        # Apply the cleaning process to each signal in each channel
        for i in range(padding_samples, data_padded.shape[2] - padding_samples):
            # Compute the start and end of the window
            start = i - padding_samples
            end = i + padding_samples

            # Identify if this sample is in a window that should be zeroed out
            mask = np.logical_not(np.any(indices_list[:, :, start:end], axis=2))

            # Apply the mask
            data_clean[:, :, i][~mask] = np.nan

        # Remove padding
        self.data = data_clean[:, :, padding_samples:-padding_samples]

    def band_pass_signal(self):
        """
        Low-pass filter procedure:
        """
        if self.method != 'cortical':
            raise MissingAttributeError('Method must be set to `cortical` to call this method')

        fs_goal = self.fs_goal
        data = self.data

        """
        3. Band-pass filter signal between 1–9 Hz.
            3.1. High-pass filter: one-pass zero- phase, order 209 (212), transition width 2 Hz
            3.2. Low-pass filter: one-pass zero-phase, order 189 (188), transition width 2.2 Hz).
        """
        lowcut = 1.0
        trans_width_high = 2.0
        order_high = determine_filter_order(fs_goal, trans_width_high)

        highpass_taps = signal.firwin(  # high-pass filter coefficients
            order_high,
            lowcut,
            fs=fs_goal,
            pass_zero=False,
            window='hamming'
        )

        highcut = 9.0
        trans_width_low = 2.2
        order_low = determine_filter_order(fs_goal, trans_width_low)

        lowpass_taps = signal.firwin(  # low-pass filter coefficients
            order_low,
            highcut,
            fs=fs_goal,
            pass_zero=True,
            window='hamming'
        )

        data = signal.filtfilt(highpass_taps, 1.0, data, axis=2)
        data = signal.filtfilt(lowpass_taps, 1.0, data, axis=2)

        self.data = data

    def high_pass_signal(self):
        """
        High-pass filter at 80 Hz (one-pass zero-phase Hamming-windowed sinc FIR corrected for filter delay, order
        671 (Florine: 676), transition width 20 Hz.
        """
        if self.method != 'subcortical':
            raise MissingAttributeError('Method must be set to `subcortical` to call this method.')

        fs = self.fs
        data = self.raw_data

        # Imitate anti-aliasing filter
        cutoff_freq = 1638.0
        trans_width = 409.6
        nyq_rate = fs / 2
        order = determine_filter_order(fs, trans_width)

        freq_points = [0, cutoff_freq, cutoff_freq + trans_width, nyq_rate]
        gain_points = [1, 1, 0, 0]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            window='hamming',
            fs=fs
        )

        data = signal.filtfilt(fir_coeff, 1.0, data, axis=2)

        del cutoff_freq, trans_width, nyq_rate, order, freq_points, gain_points, fir_coeff

        # High-pass at 80
        cutoff_freq = 80.0
        trans_width = 20.0
        nyq_rate = fs / 2
        order = determine_filter_order(fs, trans_width)

        freq_points = [0, cutoff_freq - (trans_width / 2), cutoff_freq + (trans_width / 2), nyq_rate]
        gain_points = [0, 0, 1, 1]

        fir_coeff = signal.firwin2(
            numtaps=order,
            freq=freq_points,
            gain=gain_points,
            window='hamming',
            antisymmetric=False,
            fs=fs,
        )

        data = signal.filtfilt(fir_coeff, 1.0, data, axis=2)

        self.data = data

    def cut_onset(self, cut=1):
        """
        Cuts onset of my EEG.

        Parameters
        ----------
        cut : float, optional
            The amount of time to cut from each end of the sound file (in seconds). Default is 1.
        """
        if self.method == 'cortical':
            fs = self.fs_goal
        elif self.method == 'subcortical':
            fs = self.fs

        data = self.data
        samples_to_cut = cut * fs

        self.data = data[..., samples_to_cut:]

    def normalize(self):
        data = self.data
        self.data = (data - np.mean(data)) / np.std(data)


class MissingAttributeError(Exception):
    pass


def determine_filter_order(fs, trans_width, attenuation=80):
    """
    Determines the filter order based on the sampling frequency, transition width, and attenuation.

    Args:
        fs (int): Sampling frequency in Hz.
        trans_width (float): Transition width as a fraction of the Nyquist frequency.
        attenuation (float, optional): Attenuation in dB. Defaults to 80 dB.

    Returns:
        int: Filter order.
    """
    order = int((attenuation - 8) * (fs / (22 * trans_width)))
    order = order // 2 * 2 + 1 if order % 2 else order // 2 * 2 - 1
    return order
