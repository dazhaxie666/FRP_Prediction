import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    The Encoder module of a convolutional neural network that reduces 
    the spatial dimensions of the input image while increasing the depth.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # Convolutional layers of the encoder with increasing depth and decreasing spatial dimensions
        self.conv1 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to reduce dimensions
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the encoder. Applies a series of convolutions and downsampling.
        :param x: Input tensor to the encoder.
        :return: Transformed tensor with reduced spatial dimensions and increased depth.
        """
        x = F.relu(self.conv1(x))  # Apply convolution and activation function
        x = self.pool(x)  # Apply pooling to reduce dimensions
        x = F.relu(self.conv2(x))  # Further convolutions and pooling
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))  # Last convolution layer
        return x

class Decoder(nn.Module):
    """
    The Decoder module of a convolutional neural network that reconstructs 
    the image from the encoded representation back to the original dimensions.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # Transposed convolutional layers of the decoder with decreasing depth and increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(16, 1, kernel_size=1)  # Final convolution to get to the desired output channel

    def forward(self, x):
        """
        Forward pass of the decoder. Applies a series of transposed convolutions (upconvolutions) and activation functions.
        :param x: Input tensor to the decoder.
        :return: Reconstructed tensor with dimensions similar to the original input image.
        """
        x = F.relu(self.upconv1(x))  # Apply transposed convolution and activation function
        x = F.relu(self.upconv2(x))  # Further transposed convolutions and activations
        x = F.relu(self.upconv3(x))
        x = self.conv(x)  # Final convolution to get the reconstructed image
        return x
