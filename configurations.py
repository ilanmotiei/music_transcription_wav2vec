
original_musicnet_sampling_rate = 44100
sampling_rate = 16000
bins = 49  # amount of predictions in a second of audio (#'sampling_rate' samples)
pitch_classes = 128
unit_duration = 1  # duration in seconds of each unit for which we divide the audio and the labels files
num_samples_in_unit = int(unit_duration * sampling_rate)  # amount of audio samples in a unit

musicnet_data_path = '../../music-translation/musicnet'

