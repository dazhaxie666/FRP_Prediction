import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class FireDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fire_names = os.listdir(root_dir)  # 假设每个火灾是一个子目录

        self.sample_index_ranges = []
        total_samples = 0

        for fire_name in self.fire_names:
            fire_path = os.path.join(root_dir, fire_name)
            A = torch.load(os.path.join(fire_path, fire_name + '_FRP.pt'))
            B = torch.load(os.path.join(fire_path, fire_name + '_动态.pt'))

            # 确保 A 和 B 的 sequence length 一致
            if A.size(0) != B.size(0):
                raise ValueError(f"Sequence length mismatch in {fire_name}: A({A.size(0)}), B({B.size(0)})")

            num_samples = A.size(0) - 5
            self.sample_index_ranges.append((total_samples, total_samples + num_samples))
            total_samples += num_samples

        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 找到包含 idx 的火灾事件
        for fire_idx, (start, end) in enumerate(self.sample_index_ranges):
            if start <= idx < end:
                sample_idx = idx - start  # 本地样本索引
                break

        fire_name = self.fire_names[fire_idx]
        fire_path = os.path.join(self.root_dir, fire_name)


        A = torch.load(os.path.join(fire_path, fire_name + '_FRP.pt'))
        B = torch.load(os.path.join(fire_path, fire_name + '_动态.pt'))
        C = torch.load(os.path.join(fire_path, fire_name + '_静态.pt'))
        A = torch.log(A + 1)  #对数变换

        
        A_input = A[sample_idx:sample_idx + 5, :, :, :]
        A_target = A[sample_idx + 5, :, :, :]
        B_input = B[sample_idx:sample_idx + 5, :, :, :]
        C_input = C
        
        return A_input, B_input, C_input,A_target
torch.manual_seed(0)
dataset = FireDataset(root_dir='path_to_your_data')

# 计算每个部分的大小
total_size = len(dataset)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.1)
test_size = total_size - train_size - val_size

# 将数据集分割为训练、验证和测试集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
