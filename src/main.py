import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA

# we set a torch seed for reproducability when drawing from the distribution in the VAE
torch.manual_seed(0) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # load the data
    data = np.load(DATA["corrupted"])

    # create normalized data tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28) / 255
    # create dataloader
    dataloader = DataLoader(data_tensor, batch_size=32, shuffle=False)

