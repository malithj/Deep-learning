import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TextMappingDataset(Dataset):
    def __init__(self, english_file, foriegn_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.english_txt = pd.read_csv(english_file, delimiter='\n')
        self.foriegn_txt = pd.read_csv(foriegn_file, delimiter='\n')
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        print("length: ", len(self.english_txt))
        return len(self.english_txt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _english_txt = self.english_txt.iloc[idx]
        _foriegn_txt = self.foriegn_txt.iloc[idx]
        sample = {'english_txt': _english_txt, 'foriegn_txt': _foriegn_txt}

        if self.transform:
            sample = self.transform(sample)

        return sample
