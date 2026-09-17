import torch
from torch.utils.data import Dataset
import numpy as np

class BearingDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)

        if len(self.x_data.shape) == 2:
            self.x_data = np.expand_dims(self.x_data, axis=1)
            
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.long)
        return x, y