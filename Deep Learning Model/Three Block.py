import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class StateBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, num_conv_filters, device):
        super(StateBlock, self).__init__()
        self.device = device
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=hidden_dims,
                                 kernel_size=kernel_sizes,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False).to(device)
        
        # Conv layers for processing the output of the last ConvLSTM layer
        self.conv1 = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=num_conv_filters[0], kernel_size=3, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=num_conv_filters[0], out_channels=num_conv_filters[1], kernel_size=3, padding=1).to(device)
        self.conv3 = nn.Conv2d(in_channels=num_conv_filters[1], out_channels=num_conv_filters[2], kernel_size=1, padding=0).to(device)

    def forward(self, x):
        # ConvLSTM forward
        x, _ = self.convlstm(x)
        x = x[0]  # Get the output of the last ConvLSTM layer
        x = x[:, -1, :, :, :]  # Get the last time step
        
        # Forward pass through Conv layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x
    
class DynamicBlock(nn.Module):
    def __init__(self, input_channels, convlstm_hidden_channels, conv_hidden_channels, device):
        super(DynamicBlock, self).__init__()
        self.device = device

        # 因为Dynamic Block包含4个ConvLSTM层，我们需要将hidden_channels列表扩展到4个元素
        # 假设每层ConvLSTM的隐藏状态通道数相同
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
    def __init__(self, input_channels):
        super(ConstantBlock, self).__init__()

        # 定义三个卷积层，每层都使用3x3的卷积核，步长为1，并保持输出尺寸不变
        # 输出通道数分别为8, 16, 16
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

    def forward(self, x):
        # 顺序通过三个卷积层，并在每层后应用ReLU激活函数
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x
    


    



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model1 = StateBlock(input_dim=1, hidden_dims=[4, 8, 16, 16], kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
#                    num_layers=4, num_conv_filters=[8, 16, 16], device=device)
# model2 = DynamicBlock(input_channels=9, 
#                      convlstm_hidden_channels=16,  # Assuming 16 hidden channels for ConvLSTM layers
#                      conv_hidden_channels=[8, 16, 16],  # Output channels for subsequent conv layers
#                      device=device)
# model3 = ConstantBlock(input_channels=6).to(device)
# # Create a sample input tensor 
# input_tensor = torch.rand(1, 6,9, 64, 64).to(device)
# # Perform a forward pass through the model
# output = model2(input_tensor)
# output.shape    
    
    
    
    
