import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
import torch.nn.functional as F

# Determine the device to run the network on (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StateBlock(nn.Module):
    """
    StateBlock for processing sequential data using ConvLSTM and Conv layers.
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, num_conv_filters, device):
        super(StateBlock, self).__init__()
        self.device = device
        # Initialize ConvLSTM layers
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=hidden_dims,
                                 kernel_size=kernel_sizes,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False).to(device)
        
        # Initialize Conv layers for processing the output of the last ConvLSTM layer
        self.conv1 = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=num_conv_filters[0], kernel_size=3, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=num_conv_filters[0], out_channels=num_conv_filters[1], kernel_size=3, padding=1).to(device)
        self.conv3 = nn.Conv2d(in_channels=num_conv_filters[1], out_channels=num_conv_filters[2], kernel_size=1, padding=0).to(device)

    def forward(self, x):
        # Forward pass through ConvLSTM layers
        x, _ = self.convlstm(x)
        # Select the output of the last ConvLSTM layer
        x = x[0]  # Assuming return_all_layers is False
        # Select the last time step
        x = x[:, -1, :, :, :]
        
        # Forward pass through Conv layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

class DynamicBlock(nn.Module):
     """
    DynamicBlock for processing sequential data using ConvLSTM and Conv layers.
    """
    def __init__(self, input_channels, convlstm_hidden_channels, conv_hidden_channels, device):
        super(DynamicBlock, self).__init__()
        self.device = device
        hidden_dims = [convlstm_hidden_channels] * 4

        # ConvLSTM layers
        self.convlstm = ConvLSTM(input_dim=input_channels,
                                 hidden_dim=hidden_dims,
                                 kernel_size=[(5, 5), (3, 3), (3, 3), (1, 1)],
                                 num_layers=4,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False).to(device)

        # Convolutional layers for processing the output of the last ConvLSTM layer
        self.conv1 = nn.Conv2d(in_channels=convlstm_hidden_channels, out_channels=conv_hidden_channels[0], kernel_size=3, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=conv_hidden_channels[0], out_channels=conv_hidden_channels[1], kernel_size=3, padding=1).to(device)
        self.conv3 = nn.Conv2d(in_channels=conv_hidden_channels[1], out_channels=conv_hidden_channels[2], kernel_size=1, padding=0).to(device)

    def forward(self, x):
        # ConvLSTM forward pass
        x, _ = self.convlstm(x)
        x = x[0]  # Get the output of the last ConvLSTM layer
        x = x[:, -1, :, :, :]  # Select the output of the last timestep
        
        # Forward pass through Conv layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        return x
class ConstantBlock(nn.Module):
     """
   ConstantBlock for processing spatial data using Conv layers.
    """
    def __init__(self, input_channels):
        super(ConstantBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

# Testing routine
def test_blocks():
    # Instantiate the blocks
    state_block = StateBlock(input_dim=1, hidden_dims=[4, 8, 16, 16], kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
                             num_layers=4, num_conv_filters=[8, 16, 16], device=device)
    dynamic_block = DynamicBlock(input_channels=9, convlstm_hidden_channels=16, conv_hidden_channels=[8, 16, 16], device=device)
    constant_block = ConstantBlock(input_channels=6).to(device)
    
    # Create a sample input tensor
    input_tensor = torch.rand(1, 1, 64, 64).to(device)  # Adjust dimensions as needed
    # Perform a forward pass through the state block
    state_output = state_block(input_tensor)
    print(f"Output shape from StateBlock: {state_output.shape}")
    
    # Assuming the input tensor for DynamicBlock has different dimensions
    input_tensor_dynamic = torch.rand(1, 9, 64, 64).to(device)  # Adjust dimensions as needed
    # Perform a forward pass through the dynamic block
    dynamic_output = dynamic_block(input_tensor_dynamic)
    print(f"Output shape from DynamicBlock: {dynamic_output.shape}")
    
    # Perform a forward pass through the constant block
    constant_output = constant_block(input_tensor)
    print(f"Output shape from ConstantBlock: {constant_output.shape}")

# Run the test
test_blocks()
