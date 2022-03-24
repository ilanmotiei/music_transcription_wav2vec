import librosa
import os
import configurations as cnf
import soundfile as sf
from math import floor
import pandas as pd
from collections import defaultdict
import torch
import torchaudio
import tqdm


def resample_dataset(root_dir):
    """
    Resamples all the files at the 'root_dir' to the sample rate defined at the configurations.py file.
    """

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = f'{subdir}/{file}'
            resampled_filepath = f"{subdir}/{file.split('.')[0]+'_resampled.wav'}"

            resampled_data, _ = librosa.load(filepath, sr=cnf.sampling_rate)

            sf.write(file=resampled_filepath,
                     data=resampled_data,
                     samplerate=cnf.sampling_rate)


def load_audio(filepath, torch_resampler):
    waveform = torch_resampler(torchaudio.load(filepath)[0]).squeeze(0)

    waveform_duration = waveform.size(0) / cnf.sampling_rate  # duration of the audio file in seconds
    num_units = floor(waveform_duration / cnf.unit_duration)

    unit_waves = []

    for unit_idx in range(num_units):
        unit_start = unit_idx * cnf.num_samples_in_unit  # measured by amount of audio samples
        unit_end = (unit_idx + 1) * cnf.num_samples_in_unit  # measured by amount of audio samples

        unit_waves.append(waveform[unit_start: unit_end])

    return unit_waves


def preprocess_df(df, units_in_file):
    """
    :param df: A pandas.DataFrame containing labels for a file.
    :param units_in_file: The amount of units at the corresponding audio file after we've cut it to have an
                          integer length (in seconds).

    :return: A list of dataframes. Each dataframe corresponds to a second of audio at the corresponding wav file.
             Each df contains all the information regarding its corresponding second of audio, i.e. all the notes that
             were played during that second and their value (pitch level).
    """

    seconds_dicts = [defaultdict(lambda: []) for _ in range(units_in_file)]

    for i, row in df.iterrows():
        absolute_start_time = row['start_time'] / cnf.original_musicnet_sampling_rate
        absolute_end_time = row['end_time'] / cnf.original_musicnet_sampling_rate
        pitch = row['note']

        start_unit_idx = floor(absolute_start_time / cnf.unit_duration)
        end_unit_idx = floor(absolute_end_time / cnf.unit_duration)
        
        for unit_idx in range(start_unit_idx, end_unit_idx + 1):
            if unit_idx >= units_in_file:
                break  # Those labels corresponding audio unit was cut off the audio file becasue it is the last one

            seconds_dicts[unit_idx]['absolute_start_time'].append(absolute_start_time)
            seconds_dicts[unit_idx]['absolute_end_time'].append(absolute_end_time)
            seconds_dicts[unit_idx]['note'].append(pitch)
            
    seconds_dfs = []
            
    for second_dict in seconds_dicts:
        seconds_dfs += [pd.DataFrame.from_dict(second_dict)]

    return seconds_dfs


def labels_df_to_tensor(labels_df, absolute_start_time):
    """
    :param absolute_start_time: of that unit at the audio file it is part of.
    :param labels_df: A pandas.DataFrame containing labels for a corresponding unit of an audio file,
                      containing the following columns:
                        * absolute_start_time
                        * absolute_end_time
                        * note

    :return: A tensor of shape (cnf.bins, cnf.pitch_classes) of 0-1 values indicating which notes were played at each
             bin and what are their values.
    """

    t = torch.zeros(size=(cnf.bins, cnf.pitch_classes), dtype=torch.bool)
    bin_duration = cnf.unit_duration / cnf.bins  # The amount of time (in seconds) that each bin equals to.

    for bin in range(cnf.bins):
        bin_start_time = bin_duration * bin + absolute_start_time
        bin_end_time = bin_duration * (bin + 1) + absolute_start_time

        for _, row in labels_df.iterrows():
            if (bin_start_time >= row['absolute_start_time']) and (bin_end_time <= row['absolute_end_time']):
                # The row contains a note that was played during that bin
                pitch_level = int(row['note'])
                pitch_class = pitch_level - 1

                t[bin, pitch_class] = True

    return t


def load_dataset(root_dir, train):

    # data = dict()  # of the form {(id, unit_number): (wave_tensor, label_dataframe)}
    data = defaultdict(lambda: [])  # of the form {id: (all_units_audio_tensor, list of unit_labes_dfs)}
    # all_units_audio_tensor.shape = (num_units_in_file, cnf.unit_duration * cnf.sampling_rate)

    torch_audio_resampler = torchaudio.transforms.Resample(cnf.original_musicnet_sampling_rate,
                                                           cnf.sampling_rate,
                                                           dtype=torch.float32)

    data_dir = root_dir + '/train_data' if train else root_dir + '/test_data'
    labels_dir = root_dir + '/train_labels' if train else root_dir + '/test_labels'

    relevant_files = [filename for filename in os.listdir(data_dir) if len(filename.split('.')[0].split('_')) == 1]
    # ^ : ignoring the files at the dataset that are translated versions (created with the music translation network)

    for file in tqdm.tqdm(relevant_files):
        file_id = int(file.split('.')[0])

        audio_filepath = f'{data_dir}/{file}'
        labels_filepath = f'{labels_dir}/{file_id}.csv'

        units_audios = load_audio(audio_filepath, torch_resampler=torch_audio_resampler)
        # ^ : a list of tensors each represents a unit

        num_units_in_file = len(units_audios)
        units_labels_dfs = preprocess_df(df=pd.read_csv(labels_filepath), units_in_file=num_units_in_file)

        for unit_idx, (unit_audio, unit_labels) in enumerate(zip(units_audios, units_labels_dfs)):
            # data[(int(file_id), unit_idx)] = (unit_audio, unit_labels)
            data[file_id].append((unit_audio, unit_labels))

        data[file_id] = (torch.stack([audio for audio, _ in data[file_id]]), [unit_labels_df for _, unit_labels_df in data[file_id]])

    return data


if __name__ == "__main__":
    test_data = load_dataset('../../music-translation/musicnet', train=False)

    sample_file_id = 2191
    sample_unit_idx = 1
    sample_audio, sample_labels = test_data[(sample_file_id, sample_unit_idx)]

    print(labels_df_to_tensor(sample_labels, absolute_start_time=sample_unit_idx * cnf.unit_duration))
