# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:27:12 2024

@author: 27159
"""
import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from ConvLSTM import ConvLSTM
import torch.nn.functional as F
from encoderdecoder import Encoder ,Decoder ,CombinedBlock,FullNetwork
from block import StateBlock , DynamicBlock , ConstantBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化StateBlock
block1 = StateBlock(input_dim=1,
                    hidden_dims=[4, 8, 16, 16],
                    kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
                    num_layers=4,
                    num_conv_filters=[8, 16, 16],
                    device=device)

# 实例化DynamicBlock
block2 = DynamicBlock(input_channels=9,
                      convlstm_hidden_channels=16,  # 假设ConvLSTM层的隐藏通道数为16
                      conv_hidden_channels=[8, 16, 16],  # 后续卷积层的输出通道数
                      device=device)

# 实例化ConstantBlock
block3 = ConstantBlock(input_channels=6).to(device)  # 假设输入通道数为6
combined_block=CombinedBlock(block1, block2, block3).to(device)
# 如果Encoder和Decoder类也已经定义好了，那么您可以继续实例化它们：
encoder = Encoder().to(device)  # 将Encoder实例化并移动到设备上
decoder = Decoder().to(device) 
full_network = FullNetwork(combined_block, encoder, decoder).to(device)

# 创建三个输入张量
input1 = torch.rand(1,6, 1, 64, 64).to(device)  # 假设输入形状符合block1的期望
input2 = torch.rand(1,6,9, 64, 64).to(device)  # 假设输入形状符合block2的期望
input3 = torch.rand(1, 6, 64, 64).to(device)  # 假设输入形状符合block3的期望

# 通过模型执行前向传播
output = full_network(input1, input2, input3)
print(output.shape)  # 预期输出形状是[1, 1, 64, 64]