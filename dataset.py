
from torch.utils.data import Dataset
from preprocess import load_dataset, labels_df_to_tensor
import configurations as cnf
from collections import defaultdict


class MusicNetDataset(Dataset):

    def __init__(self, train):
        super().__init__()

        self.train = train
        self.data = load_dataset(root_dir=cnf.musicnet_data_path, train=self.train)
        # ^ : a dictionary of the form {(file_id, unit_idx): (unit_audio, unit_labels_df)}

        files_units = defaultdict(lambda: 0)  # a dictionary of the form {file_id: <amount of units in that file>}
        for file_id, _ in self.data.keys():
            files_units[file_id] += 1

        self.index_to_key = {i: (file_id, unit_idx) for i, (file_id, unit_idx) in enumerate(self.data.keys())}

    def __getitem__(self, item):
        file_id, unit_idx = self.index_to_key[item]
        unit_audio, unit_label_df = self.data[(file_id, unit_idx)]
        unit_label_tensor = labels_df_to_tensor(labels_df=unit_label_df,
                                                absolute_start_time=unit_idx * cnf.unit_duration)

        return unit_audio, unit_label_tensor
