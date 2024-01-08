import torch
import torch.nn as nn
import torch.nn.functional as F

def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

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

