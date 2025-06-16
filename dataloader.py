from signal_functions import edf_spd
from pathinator import *
from torch.utils.data import Dataset

class SPDataset(Dataset):
    def __init__(self, edf_files, seconds=None, useless_channels=None):
        self.edf_files = edf_files
        self.seconds = seconds
        self.useless_channels = useless_channels

    def __len__(self):
        return len(self.edf_files)

    def __getitem__(self, idx):
        edf_path = self.edf_files[idx]
        spd_matrix = edf_spd(edf_file=edf_path, seconds=self.seconds, useless_channels=self.useless_channels).squeeze(0)
        return spd_matrix.float()  



