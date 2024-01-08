# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:14:33 2024

@author: 27159
"""
import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder layers according to the provided image
        self.conv1 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        # Applying convolutions and max pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder layers according to the provided image
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Applying transposed convolutions
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = self.conv(x)
        return x

class CombinedBlock(nn.Module):
    def __init__(self, block1, block2, block3):
        super(CombinedBlock, self).__init__()
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3

    def forward(self, x1, x2, x3):
        # Forward pass through each individual block with its respective input
        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block3(x3)
        
        # Concatenate the outputs along the channel dimension
        combined = torch.cat((out1, out2, out3), dim=1)
        return combined

# 在FullNetwork中使用CombinedBlock
class FullNetwork(nn.Module):
    def __init__(self, combined_block, encoder, decoder):
        super(FullNetwork, self).__init__()
        self.combined_block = combined_block
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x1, x2, x3):
        # Forward pass through the combined block with three separate inputs
        combined_x = self.combined_block(x1, x2, x3)
        
        # Forward pass through the encoder and decoder with the concatenated output
        encoded_x = self.encoder(combined_x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x

