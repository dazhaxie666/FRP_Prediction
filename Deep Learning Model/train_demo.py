import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
from MultimodalEncoder import StateBlock, DynamicBlock, ConstantBlock, CombinedBlock  
from BackboneEncoder import Encoder  
from BackboneDecoder import Decoder 
import torch.nn.functional as F
from loss_function import CombinedLoss

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
model = FullNetwork(combined_block, encoder, decoder).to(device)  # Move the Full Network to the computation device


criterion = CombinedLoss(w1=0.6)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 1000 # 
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_num, (A_input, B_input, C_input, A_target) in enumerate(dataloader, 1):
        A_input, B_input, C_input, A_target = A_input.to(device), B_input.to(device), C_input.to(device), A_target.to(device)
        optimizer.zero_grad()
        outputs = model(A_input, B_input, C_input)
        loss = criterion(outputs, A_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_num % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_num}/{len(dataloader)}], Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}')
