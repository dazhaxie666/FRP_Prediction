import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
from MultimodalEncoder import StateBlock, DynamicBlock, ConstantBlock, CombinedBlock  
from BackboneEncoder import Encoder  
from BackboneDecoder import Decoder 
import torch.nn.functional as F

# Set the device for computation - GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize StateBlock - processes sequential data using ConvLSTM and Conv layers
block1 = StateBlock(input_dim=1,
                    hidden_dims=[4, 8, 16, 16],  # Hidden dimensions for each ConvLSTM layer
                    kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],  # Kernel sizes for each ConvLSTM layer
                    num_layers=4,  # Number of ConvLSTM layers
                    num_conv_filters=[8, 16, 16],  # Output channels for subsequent conv layers
                    device=device)  # The computation device

# Initialize DynamicBlock - similar to StateBlock but might have different structure or dynamics
block2 = DynamicBlock(input_channels=9,  # Number of input channels
                      convlstm_hidden_channels=16,  # Hidden channels for ConvLSTM layers
                      conv_hidden_channels=[8, 16, 16],  # Output channels for subsequent conv layers
                      device=device)  # The computation device

# Initialize ConstantBlock - a block with constant or static features extraction capability
block3 = ConstantBlock(input_channels=6).to(device)  # Assuming input channel is 6

# Combine the three blocks into a single CombinedBlock
combined_block = CombinedBlock(block1, block2, block3).to(device)

# Initialize the encoder part of the network - reduces spatial dimensions
encoder = Encoder().to(device)  # Move the Encoder to the computation device

# Initialize the decoder part of the network - reconstructs the image back to original dimensions
decoder = Decoder().to(device)  # Move the Decoder to the computation device

# Define the full network by combining the blocks with the encoder and decoder
full_network = FullNetwork(combined_block, encoder, decoder).to(device)  # Move the Full Network to the computation device
