import os
import numpy as np
import mne

from SignalProcessor import SoundProcessor, NeuralProcessor
from helpers import audio_snips, subjects, bad_channels_dict

mne.set_log_level('ERROR')


def extract_neural_signal(subjects, bad_channels_dict, cluster=['eeg']):
    """
    The function reads the EEG epochs data from my audiobook experiment. It removes bad channels, sets the reference to
    the average mastoid-channels, applies ICA to auto-detect eye blinks and then interpolates bad channels. It then
    applies a baseline correction of 200 ms prior to speech onset and crops the baseline period. Finally the processed
    EEG is written to a .npy-file.

    *EEG*: My EEG data is stored in .fif files. They were segmented to speech onsets (accounted for trigger delay) and
    re-sampled from 16384 to 4096 Hz (standard MNE anti-aliasing filter). No further preprocessing was applied, but I
    identified bad channels and stored them in `bad_channels_dict`.

    Parameters
    ----------
    subjects : list
        List of subject IDs.
    bad_channels_dict : dict
        Dictionary mapping subject IDs to their corresponding bad channels.
    cluster : list, optional
        List of channel types to pick. By default, only 'eeg' is selected.

    Returns
    -------
    None
        The function doesn't return anything but writes processed EEG data to disk.
    """
    # Define directory paths
    segments_folder = '/Volumes/NeuroSSD/segments/'
    cache_folder = '/Volumes/NeuroSSD/Midcortex/cleaned_arrays/'
    os.makedirs(cache_folder, exist_ok=True)

    # Channel lists
    mastoid_channels = ['EXG3', 'EXG4']
    eog_channels = ['EXG1', 'EXG2', 'Fp1', 'Fp2']

    # Loop over subjects
    for subject_id in subjects:
        # Define file path
        fif_filename = os.path.join(segments_folder, subject_id, f'{subject_id}_audiobook_epo.fif')
        # Read epochs
        epochs = mne.read_epochs(fif_filename)

        # Remove bad channels
        epochs.info['bads'] = bad_channels_dict[subject_id]

        # Set reference to average mastoids
        epochs.set_eeg_reference(ref_channels=mastoid_channels)

        # Create a downsampled, high-pass filtered copy for ICA
        epochs_ica_filter = epochs.copy().resample(epochs.info['sfreq'] / 4)
        epochs_ica_filter.filter(l_freq=1.0, h_freq=None)

        # Run ICA on the filtered epochs copy
        ica = mne.preprocessing.ICA(
            n_components=0.999,
            method='picard',
            max_iter=1000,
            fit_params=dict(fastica_it=5),
            random_state=51222
        )

        ica.fit(epochs_ica_filter)

        # Auto-identify bad EOG components and exclude in copy
        eog_inds, _ = ica.find_bads_eog(epochs_ica_filter, ch_name=eog_channels)
        ica.exclude = eog_inds

        # Apply to original epochs object
        ica.apply(epochs)

        # Interpolate bad channels
        epochs.interpolate_bads(reset_bads=True)

        # Empty memory
        del epochs_ica_filter, ica

        # Apply baseline correction, then cut baseline period off
        epochs.apply_baseline(baseline=(-0.200, None))
        epochs.crop(tmin=0, tmax=None)

        # Cache EEG data as numpy array
        npy_file = epochs.get_data(picks=cluster)
        npy_filename = os.path.join(cache_folder, f'{subject_id}.npy')
        np.save(npy_filename, npy_file)

        print(f'{subject_id} EEG array cached.')

    print('All EEG arrays were cached.')


def preprocess_speech(audio_snips, method):
    """
    The function applies my filtering pipeline that preprepares my data for encoder modeling to the audio files. I
    defined my filternig techniques in the SignalProcessor module. The pipeline and differs according to the specified
    method `cortical` or `subcortical`. The data is then saved to disk in .npy format.

    Parameters
    ----------
    audio_snips : list
        List of audio snippet IDs.
    method : str
        Processing method. Must be either `cortical` or `subcortical.

    Returns
    -------
    None
        The function doesn't return anything but writes processed audio data to disk.
    """
    # Define processing parameters and file path based on method
    if method == 'cortical':
        fs = 128
        npy_file = 'audio/low_envelopes.npy'
    elif method == 'subcortical':
        fs = 4096
        npy_file = 'audio/rectified_audios.npy'
    else:
        raise ValueError('Invalid method argument. Valid options are `cortical` or `subcortical`.')

    eeg_length = int(50 * fs)  # 50 seconds

    # Initialize a NaN-filled array with shape for ReceptiveField estimator (n_times, n_epochs, n_channels)
    all_data = np.full((eeg_length, len(audio_snips)), np.nan)

    # Process each audio snippet based on the method
    for idx, snip_id in enumerate(audio_snips):
        sound_processor = SoundProcessor(snip_id, method=method)
        sound_processor.remove_silent_onset()

        if method == 'cortical':
            sound_processor.extract_gammatone_envelope()
            sound_processor.low_pass_signal()
            sound_processor.cut_ends(cut=1)
            signal = sound_processor.low_passed_signal
        elif method == 'subcortical':
            sound_processor.high_pass_signal()
            sound_processor.cut_ends(cut=1)
            signal = sound_processor.high_passed_signal

        # Pad the signal with NaNs to match the length of EEG data
        padding_length = eeg_length - signal.shape[0]
        signal = np.pad(signal, (0, padding_length), constant_values=np.nan)

        all_data[:, idx] = signal

    # Normalize data, replace NaNs with zeros, and add a new axis for the number of audio channels
    normalized_data = (all_data - np.nanmean(all_data)) / np.nanstd(all_data)
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    normalized_data = normalized_data[..., np.newaxis]

    # Create the directory if it doesn't exist
    os.makedirs('audio', exist_ok=True)

    # Save the processed audio data to disk
    np.save(npy_file, normalized_data)

    print(f'All speech data for {method} encoder were cached.')


def preprocess_eeg(subjects, method):
    """
    The function applies my filtering pipeline that preprepares my data for encoder modeling to the eeg files. I defined
    my filternig techniques in the SignalProcessor module. The pipeline and differs according to the specified method
    `cortical` or `subcortical`. The data is then saved to disk in .npy format.

    Parameters
    ----------
    subjects : list
        List of subject IDs.
    method : str
        Processing method. Must be either 'cortical' or 'subcortical'.

    Returns
    -------
    None
        The function doesn't return anything but writes processed EEG data to disk.
    """
    # Define the directory path
    folder = '/Volumes/NeuroSSD/Midcortex/'

    # Loop over subjects
    for subject_id in subjects:
        # Process EEG data based on the specified method
        if method == 'cortical':
            neural_processor = NeuralProcessor(subject_id, method='cortical')
            neural_processor.low_pass_signal()
            out_folder = f'{folder}/cortex_encoder'
        elif method == 'subcortical':
            neural_processor = NeuralProcessor(subject_id, method='subcortical')
            neural_processor.high_pass_signal()
            out_folder = f'{folder}/subcortex_encoder'
        else:
            raise ValueError('Invalid method argument. Valid options are `cortical` or `subcortical`.')

        # Cut the onset and normalize the signal
        neural_processor.cut_onset(cut=1)
        neural_processor.normalize()
        eeg = neural_processor.normalized_eeg

        # Transpose the data to the shape required for ReceptiveField estimator (n_times, n_epochs, n_channels)
        eeg = eeg.transpose((2, 0, 1))

        # Create the directory if it doesn't exist
        os.makedirs(out_folder, exist_ok=True)

        # Save the processed EEG data to disk
        npy_filename = os.path.join(folder, out_folder, f'{subject_id}.npy')
        np.save(npy_filename, eeg)

        print(f'{subject_id} encoder data cache')

    print(f'All EEG data for {method} encoder were cached.')


if __name__ == '__main__':
    # Speech processing
    print('Speech signal processing.')
    preprocess_speech(audio_snips, method='cortical')
    preprocess_speech(audio_snips, method='subcortical')
    print('--------------------')

    # EEG extraction
    print('EEG signal exctraction.')
    extract_neural_signal(subjects, bad_channels_dict)
    print('--------------------')

    # EEG preprocessing
    preprocess_eeg(subjects, method='cortical')
    preprocess_eeg(subjects, method='subcortical')
    print('--------------------')
