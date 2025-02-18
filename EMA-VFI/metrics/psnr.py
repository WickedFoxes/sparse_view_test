import torch

def psnr(img_1: torch.Tensor, img_2: torch.Tensor)->float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.

    Returns:
        float: PSNR result.
    """
    
    assert img_1.shape == img_2.shape, (f'Image shapes are different: {img_1.shape}, {img_2.shape}.')
    assert len(img_1.shape) == 3, 'The input should be 3D tensor'
    assert img_1.min() >= 0 and img_1.max() <= 1, 'The value of img should be in [0, 1]'
    assert img_2.min() >= 0 and img_2.max() <= 1, 'The value of img2 should be in [0, 1]'

    img_1 = img_1.to(torch.float64)
    img_2 = img_2.to(torch.float64)

    mse = torch.mean((img_1 - img_2)**2)
    return (10. * torch.log10(1. / (mse + 1e-8))).item()

if __name__ == '__main__':
    # Example
    torch.manual_seed(0)
    img_1, img_2 = torch.rand(3, 256, 256), torch.zeros(3, 256, 256)
    print(psnr(img_1, img_2))
