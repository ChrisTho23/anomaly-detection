import torch
import torch.optim as optim

from config import MODEL
from model import Encoder, Decoder, VAE, vae_loss
from preprocessing import preprocessing

if __name__ == "__main__":
    # we set a torch seed for reproducability when drawing from the distribution in the VAE
    torch.manual_seed(0) 
    # select GPU for processing if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the data
    dataloader = preprocessing()

    # instantiate the encoder and decoder models
    encoder = Encoder(MODEL["image_size"], MODEL["embedding_dim"]).to(device)
    decoder = Decoder(MODEL["embedding_dim"], (128, 4, 4)).to(device)
    # pass the encoder and decoder to VAE class
    model = VAE(encoder, decoder)

    # instantiate optimizer and scheduler
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters())
    )

    print("Start VAE training")

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

    print("Finished VAE training")

    # save model
    torch.save(model, MODEL["vae"])

