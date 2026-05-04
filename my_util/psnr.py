import torch
import torch.nn.functional as F

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 255.0) -> float:
    # 确保输入张量的形状相同
    if img1.shape != img2.shape:
        raise ValueError("Input tensors must have the same shape.")

    diff = img1 - img2
    mse = torch.sqrt(torch.mean(diff**2))
    # 计算 PSNR
    if mse.item() == 0: # 如果 RMSE 为 0，表示两张图像完全相同
        psnr = 100.0
    else:
        psnr = 20 * torch.log10(255.0 / mse)
    return psnr
    # # 计算均方误差 (MSE)
    # mse = F.mse_loss(img1, img2)  # 使用 PyTorch 的均方误差损失函数
    # if mse == 0:  # 如果 MSE 为 0，表示两张图像完全相同
    #     return float('inf')  # PSNR 为无穷大
    #
    # # 计算 PSNR
    # psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    # return psnr  # 返回 PSNR 值
