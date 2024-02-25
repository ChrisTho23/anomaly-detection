import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA, MODEL

def preprocessing():
    """Preprocess the data by reading the corrupted data and creating a dataloader.

    Returns:
        dataloader (DataLoader): dataloader for the corrupted data
    """
    # load the data
    data = np.load(DATA["corrupted"])

    # create normalized data tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).view(
        -1, 1, MODEL["image_size"], MODEL["image_size"]
    ) / 255
    # create dataloader
    dataloader = DataLoader(data_tensor, batch_size=32, shuffle=False)

    return dataloader