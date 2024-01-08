import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        """
        Initialize ConvLSTM.

        Parameters:
        input_dim (int): Number of channels of input tensor.
        hidden_dim (list): Number of channels of hidden states for each layer.
        kernel_size (list): Size of the convolutional kernel for each layer.
        num_layers (int): Number of ConvLSTM layers.
        batch_first (bool): Whether or not the input tensor's first dimension is batch size.
        bias (bool): Whether or not to add the bias.
        return_all_layers (bool): Whether to return the outputs of all layers or only the last layer.
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists of the same length as `num_layers`
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create a list of ConvLSTM cells for each layer
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass of ConvLSTM.

        Parameters:
        input_tensor (Tensor): Input tensor, 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state (list): List of hidden states for each layer. If None, they will be initialized to zeros.

        Returns:
        list: Outputs and last states of each layer or only the last layer depending on `return_all_layers`.
        """
        if not self.batch_first:
            # Convert input from shape (t, b, c, h, w) to (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),
                                             image_size=(input_tensor.size(3), input_tensor.size(4)))

        # Forward pass for each layer
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # Return outputs of all layers or only the last layer
        return layer_output_list if self.return_all_layers else layer_output_list[-1], \
               last_state_list if self.return_all_layers else last_state_list[-1]

    # Rest of the class methods...

# Test routine for ConvLSTM
def test_convlstm():
    # Example input - batch size of 1, sequence length of 5, 3 channels, 64x64 image
    input_tensor = torch.rand(1, 5, 3, 64, 64)  
    # Example ConvLSTM with single layer
    convlstm = ConvLSTM(input_dim=3, hidden_dim=[16], kernel_size=[(3, 3)], num_layers=1)
    convlstm.to('cpu')  # Move ConvLSTM to CPU (or GPU if available)
    # Initialize hidden state
    hidden_state = convlstm._init_hidden(batch_size=1, image_size=(64, 64))
    # Forward pass
    output, last_state = convlstm(input_tensor, hidden_state)
    print(f"Output shape: {output[0].shape}")
    print(f"Last hidden state shape: {last_state[0][0].shape}")
    print(f"Last cell state shape: {last_state[0][1].shape}")

# Run the test
test_convlstm()
