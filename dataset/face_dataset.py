import torch
from torch.utils.data import Dataset
import numpy as np
import pandas

from .utils import load_raw_data
from .transforms import Compose 

class FaceDataset(Dataset):

    def __init__(self, data, labels=None, transforms=None):
        self.data = torch.tensor(data, dtype=torch.float)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.labels = None
        self.transforms = Compose(transforms) 

    def __getitem__(self, index):
        if self.labels is not None:
            x, y = self.data[index], self.labels[index]
            x, y = self.transforms(x, y)
            return x, y
        else: 
            x = self.data[index]
            return x
        
    def __len__(self):
        return len(self.data)