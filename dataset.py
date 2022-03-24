
import torch
from torch.utils.data import Dataset
from preprocess import load_dataset, labels_df_to_tensor
import configurations as cnf
from collections import defaultdict


class MusicNetDataset(Dataset):

    def __init__(self, train):
        super().__init__()

        self.train = train
        self.data = load_dataset(root_dir=cnf.musicnet_data_path, train=self.train)
        # ^ : a dictionary of the form {file_id: (all_units_audio_tensor, list_of_unit_labels_df)}
        # all_units_audio_tensor.shape = (num_units_in_file, cnf.unit_duration * cnf.sampling_rate)
        # len(list_of_unit_labels_df) = num_units_in_file

        keys = [(file_id, j) for file_id in self.data.keys() for j in range(len(self.data[file_id][1]))]
        self.index_to_key = {i: (file_id, unit_idx) for i, (file_id, unit_idx) in enumerate(keys)}

    def __len__(self):
        return len(self.index_to_key.keys())

    def __getitem__(self, item):
        file_id, unit_idx = self.index_to_key[item]

        unit_audio = self.data[file_id][0][unit_idx]
        unit_label_df = self.data[file_id][1][unit_idx]

        unit_label_tensor = labels_df_to_tensor(labels_df=unit_label_df,
                                                absolute_start_time=unit_idx * cnf.unit_duration)

        return unit_audio, unit_label_tensor
