import torch
import cv2
import numpy as np
import torch.nn.functional as F

def _calculate_ssim_pt(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim

def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

def ssim(img_1:torch.Tensor, img_2:torch.Tensor)->float:
    """
    Calculate SSIM (structural similarity) (PyTorch version).

    Reference: https://ece.uwaterloo.ca/~z70wang/research/ssim/

    Args:
        img_1 (Tensor): Images with range [0, 1], shape (3/1, h, w).
        img_2 (Tensor): Images with range [0, 1], shape (3/1, h, w).

    Returns:
        float: SSIM result.
    """
    assert img_1.shape == img_2.shape, (f'Image shapes are different: {img_1.shape}, {img_2.shape}.')
    assert len(img_1.shape) == 3, 'The input should be 3D tensor'
    assert img_1.min() >= 0 and img_1.max() <= 1, 'The value of img_1 should be in [0, 1]'
    assert img_2.min() >= 0 and img_2.max() <= 1, 'The value of img_2 should be in [0, 1]'

    return _calculate_ssim_pt(img_1.unsqueeze(0), img_2.unsqueeze(0)).item()

if __name__ == '__main__':
    # Example
    torch.manual_seed(0)
    img_1, img_2 = torch.rand(3, 256, 256), torch.rand(3, 256, 256)
    print(ssim(img_1, img_2))