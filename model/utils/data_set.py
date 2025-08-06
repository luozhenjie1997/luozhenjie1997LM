import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset


class ABodyBuilder3Dataset(Dataset):
    def __init__(self, data_source, plm_embed_path, processed_path):
        super(ABodyBuilder3Dataset, self).__init__()
        self.df = pd.read_csv(data_source)
        self.processed_path = processed_path
        self.plm_embed = h5py.File(plm_embed_path, 'r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['id']
        data = torch.load(self.processed_path + '/%s.pt' % file_name)
        data['antibody_emb'] = torch.from_numpy(self.plm_embed[file_name][:])
        return data
