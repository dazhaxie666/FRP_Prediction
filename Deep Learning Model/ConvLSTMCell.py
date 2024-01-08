import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters:
        input_dim (int): Number of channels of input tensor.
        hidden_dim (int): Number of channels of hidden state.
        kernel_size (tuple): Size of the convolutional kernel.
        bias (bool): Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Kernel size must be a tuple
        self.kernel_size = kernel_size
        # Padding ensures the output feature map has the same size as the input
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolutional layer that combines input and previous hidden state
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # for input, forget, cell, and output gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Parameters:
        input_tensor (Tensor): Input tensor at current time-step.
        cur_state (tuple): Tuple of (h_cur, c_cur), the current hidden and cell states.

        Returns:
        tuple: Next hidden and cell states (h_next, c_next).
        """
        h_cur, c_cur = cur_state
        # Concatenate along channel axis for combined state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Pass through the convolutional layer
        combined_conv = self.conv(combined)
        # Split the combined convolution output into 4 parts for different gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # Apply sigmoid activation for input, forget, and output gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        # Apply tanh activation for candidate cell state
        g = torch.tanh(cc_g)

        # Calculate next cell state
        c_next = f * c_cur + i * g
        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize hidden state.

        Parameters:
        batch_size (int): Batch size.
        image_size (tuple): Size of the image.

        Returns:
        tuple: Initial hidden and cell states filled with zeros.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# Test routine for ConvLSTMCell
def test_convlstm_cell():
    # Sample input
    height, width = 64, 64
    input_tensor = torch.rand(1, 3, height, width)  # Batch size of 1, 3 channels, 64x64 image
    # Sample ConvLSTM cell
    convlstm_cell = ConvLSTMCell(input_dim=3, hidden_dim=16, kernel_size=(5, 5))
    convlstm_cell.to('cpu')  # Move cell to CPU (or GPU if available)
    # Initialize hidden state
    h, c = convlstm_cell.init_hidden(batch_size=1, image_size=(height, width))
    # Forward pass
    h_next, c_next = convlstm_cell(input_tensor, (h, c))
    print(f"Output shape of hidden state: {h_next.shape}")
    print(f"Output shape of cell state: {c_next.shape}")

# Run the test
test_convlstm_cell()
