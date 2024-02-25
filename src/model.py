"""File for the VAE model and loss.
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

class Sampling(nn.Module):
    """Class to sample from the latent space distribution of the VAE."""
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    """The Encoder part of the Variational Autoencoder (VAE).

    The encoder has three convolutional layers to downsample the input image.
    The output of the convolutional layers is then flattened and passed through
    two fully connected layers to get the mean and log variance of the latent
    space distribution.

    Args:
        image_size (int): Number of pixels per side in picture, assuming quadratic pictures
        embedding_dim (int): Dimension of latent space variables

    Attributes:
        conv1 (): description
        conv2 (): description
        conv3 (): description
        flatten (): description
        fc_mean (): description
        fc_log_var (): description
        sampling (Sampling): description
    """
    def __init__(self, image_size, embedding_dim):
        super(Encoder, self).__init__()
        # convolutional layers for downsampling and feature extraction
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) 
        # define a flatten layer to flatten the tensor
        self.flatten = nn.Flatten()
        # define fully connected layers to transform the tensor into embedding dimensions
        self.fc_mean = nn.Linear(
            2048, embedding_dim
        )
        self.fc_log_var = nn.Linear(
            2048, embedding_dim
        )
        # initialize the sampling layer
        self.sampling = Sampling()
    def forward(self, x):
        """Forward pass of the encoder.

        This function takes an input tensor, applies a series of convolutional 
        layers followed by ReLU activations, flattens the output, and then 
        applies two fully connected layers to produce the mean and log variance 
        of the latent space distribution. These outputs are then used to sample 
        a point in the latent space using the reparameterization trick.

        Args:
            x (torch.Tensor): The input tensor, typically an image or a batch of images.
                            The tensor should have a shape that matches the expected 
                            input size of the convolutional layers.

        Returns:
            tuple of torch.Tensor: 
                - z_mean (torch.Tensor): The mean of the latent space distribution.
                - z_log_var (torch.Tensor): The log variance of the latent space distribution.
                - z (torch.Tensor): A sampled point from the latent space.
        """
        # apply convolutional layers with relu activation function
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten the tensor
        x = self.flatten(x)
        # get the mean and log variance of the latent space distribution
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        # sample a latent vector using the reparameterization trick
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """
    The Decoder part of the Variational Autoencoder (VAE).

    Args:
        embedding_dim (int): The dimension of the latent space.
        shape_before_flattening (Tuple of int): The shape of the tensor before it was 
                                                flattened in the encoder. 
    """
    def __init__(self, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()
        # fully connected layer to transform the latent vector back to shape before flattening
        self.fc = nn.Linear(
            embedding_dim,
            shape_before_flattening[0]
            * shape_before_flattening[1]
            * shape_before_flattening[2],
        )
        self.shape_before_flattening = shape_before_flattening
        # transposed convolutional layers to upsample and generate the reconstructed image
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, 3, stride=2, padding=1, output_padding=1
        )
    def reshape_tensor(self, x):
        return x.view(-1, *self.shape_before_flattening)
    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): The latent vector, typically outputted by the encoder.

        Returns:
            torch.Tensor: The reconstructed image or data point.
        """
        # pass the latent vector through the fully connected layer
        x = self.fc(x)
        # reshape the tensor
        x = self.reshape_tensor(x) 
        # apply transposed convolutional layers with relu activation function
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # apply the final transposed convolutional layer with a sigmoid to generate the final output
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) combining both the Encoder and Decoder.

    This class defines the entire VAE architecture, which includes the encoder 
    that maps input data to a latent space, and the decoder that reconstructs 
    data from the latent space representation. 

    Attributes:
        encoder (nn.Module): The encoder part of the VAE.
        decoder (nn.Module): The decoder part of the VAE.

    The forward method of this class takes input data, encodes it into a latent 
    vector, and then decodes it back into reconstructed data.
    """
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple of torch.Tensor: 
                - tuple containing the mean and log variance 
                - the reconstructed image
        """
        # pass the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x)
        # pass the latent vector through the decoder to get the reconstructed image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction


def vae_gaussian_kl_loss(mu, logvar):
    """

    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114

    Args:
        mu (_type_): _description_
        logvar (_type_): _description_

    Returns:
        _type_: _description_
    """
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()

def reconstruction_loss(x_reconstructed, x):
    """

    Args:
        x_reconstructed (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    bce_loss = nn.BCELoss()
    return bce_loss(x_reconstructed, x)

def vae_loss(y_pred, y_true):
    """_summary_

    Args:
        y_pred (_type_): _description_
        y_true (_type_): _description_

    Returns:
        _type_: _description_
    """
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return 500 * recon_loss + kld_loss