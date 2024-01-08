import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim  # Ensure pytorch_msssim is installed for SSIM

class CombinedLoss(nn.Module):
    """
    A custom loss function that combines Mean Squared Error (MSE) and 
    Structural Similarity Index Measure (SSIM) into a single loss function.
    The loss function can be weighted to balance the contribution of each component.
    """
    def __init__(self, w1, w2=None):
        """
        Initialize the CombinedLoss module.

        Parameters:
        w1 (float): The weight for the MSE loss component.
        w2 (float, optional): The weight for the SSIM loss component. If None, it is set to (1 - w1).
        """
        super(CombinedLoss, self).__init__()
        self.w1 = w1  # Weight for MSE loss
        self.w2 = 1 - w1 if w2 is None else w2  # Weight for SSIM loss, ensuring w1 + w2 = 1
        self.mse_loss = nn.MSELoss()  # The MSE loss component

    def forward(self, y_pred, y_true):
        """
        Forward pass of the loss function.

        Parameters:
        y_pred (torch.Tensor): The predicted tensor.
        y_true (torch.Tensor): The ground truth tensor.

        Returns:
        torch.Tensor: The combined loss value.
        """
        # Calculate MSE loss
        loss_mse = self.mse_loss(y_pred, y_true)
        # Calculate SSIM loss. Assumes y_pred and y_true are normalized in the range [0, 1].
        loss_ssim = 1 - ssim(y_pred, y_true, data_range=1, size_average=True)
        # Combine the losses using the specified weights
        combined_loss = self.w1 * loss_mse + self.w2 * loss_ssim
        return combined_loss

