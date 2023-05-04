import os
import numpy as np
import mne

from SignalProcessor import SoundProcessor, NeuralProcessor
from helpers import audio_snips, subjects, auditory_cluster, bad_channels_dict


mne.set_log_level('ERROR')


def extract_neural_signal(subjects, bad_channels_dict, method, cluster=['eeg']):
    for subject_id in subjects:
        eeg = f'/Volumes/NeuroSSD/segments/{subject_id}/{subject_id}_audiobook_epo.fif'
        epochs = mne.read_epochs(eeg)

        epochs.info['bads'] = bad_channels_dict[subject_id]
        epochs.set_eeg_reference(ref_channels=['EXG3', 'EXG4'])

        if method == 'cortical':
            folder = 'cortex'

            # ICA for cortical analyses only
            epochs_ica_filter = epochs.copy().resample(epochs.info['sfreq'] / 4)
            epochs_ica_filter.filter(l_freq=1.0, h_freq=None)

            ica = mne.preprocessing.ICA(
                n_components=0.999,
                method='picard',
                max_iter=1000,
                fit_params=dict(fastica_it=5),
                random_state=51222
            )
            ica.fit(epochs_ica_filter)

            eog_inds, _ = ica.find_bads_eog(
                epochs_ica_filter,
                ch_name=['EXG1', 'EXG2']
            )

            ica.exclude = eog_inds

            # Apply to original object
            ica.apply(epochs)

            epochs.interpolate_bads(reset_bads=True)
            del epochs_ica_filter, ica

        elif method == 'subcortical':
            folder = 'subcortex'

        else:
            raise ValueError(f'`{method}` must be `cortical` or `subcortical`.')

        # Apply baseline correction, than cut to 0
        epochs.apply_baseline(baseline=(-0.200, None))
        epochs.crop(tmin=0, tmax=None)

        np.save(
            f'/Volumes/NeuroSSD/Midcortex/{folder}/{subject_id}.npy',
            epochs.get_data(picks=cluster)
        )
        print(f'{subject_id} raw {method} EEG cache.')
    print(f'All raw {method} EEG extraction data were cached.')


def preprocess_speech(audio_snips, method):
    if method == 'cortical':
        fs = 128
        npy_file = 'audio/low_envelopes.npy'
    elif method == 'subcortical':
        fs = 4096
        npy_file = 'audio/rectified_audios.npy'
    else:
        raise ValueError('Invalid method argument. Valid options are `cortical` or `subcortical`.')

    eeg_length = int(50 * fs)  # 50 seconds

    # Shape for ReceptiveField estimator must be (n_times, n_epochs, n_channels)
    all_data = np.full((eeg_length, len(audio_snips)), np.nan)

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

        # NaN padding
        padding_length = eeg_length - signal.shape[0]
        signal = np.pad(signal, (0, padding_length), constant_values=np.nan)

        all_data[:, idx] = signal

    # Normalization + padding replacement + add no. of audio channels
    normalized_data = (all_data - np.nanmean(all_data)) / np.nanstd(all_data)
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    normalized_data = normalized_data[..., np.newaxis]

    os.makedirs('audio', exist_ok=True)
    np.save(npy_file, normalized_data)
    print(f'All speech data for {method} encoder were cached.')


def preprocess_eeg(subjects, method):
    for subject_id in subjects:
        if method == 'cortical':
            neural_processor = NeuralProcessor(subject_id, method='cortical')
            neural_processor.low_pass_signal()
            npy_dir = 'eeg/cortex'
        elif method == 'subcortical':
            neural_processor = NeuralProcessor(subject_id, method='subcortical')
            neural_processor.high_pass_signal()
            npy_dir = 'eeg/subcortex'
        else:
            raise ValueError('Invalid method argument. Valid options are `cortical` or `subcortical`.')

        neural_processor.cut_onset(cut=1)
        neural_processor.normalize()
        eeg = neural_processor.normalized_eeg

        # Transpose to shape for ReceptiveField estimator (n_times, n_epochs, n_channels)
        eeg = eeg.transpose((2, 0, 1))

        os.makedirs(npy_dir, exist_ok=True)
        np.save(f'{npy_dir}/{subject_id}.npy', eeg)
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
    extract_neural_signal(subjects, bad_channels_dict, method='cortical')
    extract_neural_signal(subjects, bad_channels_dict, method='subcortical', cluster=['Pz', 'Fz', 'Cz'])
    print('--------------------')

    # EEG preprocessing
    preprocess_eeg(subjects, method='cortical')
    preprocess_eeg(subjects, method='subcortical')
    print('--------------------')
