from pytorch_msssim import ssim
import math

# PSNR function
def psnr(target, output, max_val=1.0):
    mse = torch.mean((target - output) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

# Test the model on the test dataset and calculate PSNR and SSIM
def test_model(test_loader, model, device):
    model.eval()
    total_ssim = 0
    total_psnr = 0
    with torch.no_grad():
        for A_input, B_input, C_input, A_target in test_loader:
            A_input, B_input, C_input, A_target = A_input.to(device), B_input.to(device), C_input.to(device), A_target.to(device)
            outputs = model(A_input, B_input, C_input)
            # Calculate and accumulate PSNR and SSIM
            batch_psnr = psnr(A_target, outputs)
            batch_ssim = ssim(outputs, A_target, data_range=1, size_average=True) # Ensure data_range matches your input range
            total_psnr += batch_psnr
            total_ssim += batch_ssim.item()  # .item() to get a Python number from a tensor containing a single value

    # Calculate average PSNR and SSIM over all batches in test_loader
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    print(f"Average PSNR on test set: {avg_psnr}")
    print(f"Average SSIM on test set: {avg_ssim}")

# Assuming model, test_loader and device are already defined and set up
test_model(test_loader, model, device)
