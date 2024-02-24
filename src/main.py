import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from config import DATA, MODEL
from model import Encoder, Decoder, VAE, vae_gaussian_kl_loss, vae_loss, reconstruction_loss

if __name__ == "__main__":
    # we set a torch seed for reproducability when drawing from the distribution in the VAE
    torch.manual_seed(0) 
    # select GPU for processing if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the data
    data = np.load(DATA["corrupted"])

    # create normalized data tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).view(
        -1, 1, MODEL["image_size"], MODEL["image_size"]
    ) / 255
    # create dataloader
    dataloader = DataLoader(data_tensor, batch_size=32, shuffle=False)

    # instantiate the encoder and decoder models
    encoder = Encoder(MODEL["image_size"], MODEL["embedding_dim"]).to(device)
    decoder = Decoder(MODEL["embedding_dim"], (128, 4, 4)).to(device)
    # pass the encoder and decoder to VAE class
    model = VAE(encoder, decoder)

    # instantiate optimizer and scheduler
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters())
    )

    # train model
    model.train()
    for epoch in range(5):
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = vae_loss(output, data)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.data.item()))

