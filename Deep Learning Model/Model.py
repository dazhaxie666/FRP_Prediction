import torch
import torch.nn as nn

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
