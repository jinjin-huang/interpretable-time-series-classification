import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
