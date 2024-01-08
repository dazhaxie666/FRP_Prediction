import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
"""The implementation of the DataLoader in this Python file is specifically tailored to the structure and organization of the author's dataset. It is dependent on the hierarchical arrangement and location of the data as established by the user. For use with different datasets, adjustments to the file paths, naming conventions, and data handling methods within the DataLoader might be necessary.
"""
class FireDataset(Dataset):
    """A custom Dataset class for fire data."""
    def __init__(self, root_dir):
        """
        Initialize the FireDataset.
        
        Args:
            root_dir (str): The root directory where the fire data is stored.
                            Each sub-directory represents a fire incident.
        """
        self.root_dir = root_dir
        self.fire_names = os.listdir(root_dir)  

        self.sample_index_ranges = []  # To store the index range of samples for each fire incident
        total_samples = 0  # To count the total number of samples across all incidents

        # Loop through each fire incident and load the data
        for fire_name in self.fire_names:
            fire_path = os.path.join(root_dir, fire_name)
            A = torch.load(os.path.join(fire_path, fire_name + '_FRP.pt'))  # Load FRP data
            B = torch.load(os.path.join(fire_path, fire_name + '_动态.pt'))  # Load dynamic data

            # Ensure the sequence length of A and B match
            if A.size(0) != B.size(0):
                raise ValueError(f"Sequence length mismatch in {fire_name}: A({A.size(0)}), B({B.size(0)})")

            # Calculate the number of samples for this incident and update total_samples
            num_samples = A.size(0) - 5
            self.sample_index_ranges.append((total_samples, total_samples + num_samples))
            total_samples += num_samples

        self.total_samples = total_samples  # Set the total number of samples

    def __len__(self):
        """Return the total number of samples."""
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset at the specified index.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            A tuple (A_input, B_input, C_input, A_target) for the model.
        """
        # Identify the fire incident containing the requested index
        for fire_idx, (start, end) in enumerate(self.sample_index_ranges):
            if start <= idx < end:
                sample_idx = idx - start  # Local index within the incident
                break

        # Load data for the identified fire incident
        fire_name = self.fire_names[fire_idx]
        fire_path = os.path.join(self.root_dir, fire_name)
        A = torch.load(os.path.join(fire_path, fire_name + '_FRP.pt'))  # Load FRP data
        B = torch.load(os.path.join(fire_path, fire_name + '_动态.pt'))  # Load dynamic data
        C = torch.load(os.path.join(fire_path, fire_name + '_静态.pt'))  # Load static data
        A = torch.log(A + 1)  # Apply logarithmic transformation to A

        # Extract the input and target data for the specified index
        A_input = A[sample_idx:sample_idx + 5, :, :, :]  # Input sequence of A
        A_target = A[sample_idx + 5, :, :, :]  # Target frame for A
        B_input = B[sample_idx:sample_idx + 5, :, :, :]  # Corresponding sequence of B
        C_input = C  # Static data for the incident
        
        return A_input, B_input, C_input, A_target  # Return the data tuple

# Set random seed for reproducibility
torch.manual_seed(0)
dataset = FireDataset(root_dir='path_to_your_data')

# Calculate the sizes for each dataset split
total_size = len(dataset)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.1)
test_size = total_size - train_size - val_size

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # DataLoaders are now ready to be used for training and evaluation.
