import cv2
import numpy as np
import torch
import torch.nn.functional as F

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: C x H x W, torch array
        sigma_real: C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1., dim=[-1, -2, -3])
    return distance

def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=img_noisy.shape[0]).to(img_gt.device) # win_size x win_size x 3 x 3
    sigma = gaussblur(err2, kernel, win_size, chn=img_noisy.shape[0])
    sigma.clamp_(min=1e-10)

    return sigma

def estimate_sigma_gauss_with_noise_map(err):
    win_size = 7
    err2 = (err) ** 2
    kernel = get_gausskernel(win_size, chn=err.shape[0]).to(err.device) # win_size x win_size x 3 x 3
    sigma = gaussblur(err2, kernel, win_size, chn=err.shape[0])
    sigma.clamp_(min=1e-10)

    return sigma

def akld(fake_noisy_image: torch.Tensor, real_noisy_image: torch.Tensor, clean_image: torch.Tensor)->float:
    '''
    Calculate the AKLD between fake_noisy_image and real_noisy_image
    '''
    assert len(fake_noisy_image.shape) == len(real_noisy_image.shape) == len(clean_image.shape) == 3, 'The input should be 3D tensor'
    assert fake_noisy_image.shape == real_noisy_image.shape == clean_image.shape, 'The shape of fake_noisy_image, real_noisy_image and clean_image should be the same'
    assert fake_noisy_image.min() >= 0 and fake_noisy_image.max() <= 1, 'The value of fake_noisy_image should be in [0, 1]'
    assert real_noisy_image.min() >= 0 and real_noisy_image.max() <= 1, 'The value of real_noisy_image should be in [0, 1]'
    assert clean_image.min() >= 0 and clean_image.max() <= 1, 'The value of clean_image should be in [0, 1]'
    
    sigma_real = estimate_sigma_gauss(real_noisy_image, clean_image)
    sigma_fake = estimate_sigma_gauss(fake_noisy_image, clean_image)
    return kl_gauss_zero_center(sigma_fake, sigma_real).item()

def akld_with_noise_map(fake_noise_map: torch.Tensor, real_noise_map: torch.Tensor)->float:
    '''
    Calculate the AKLD between fake_noisy_image and real_noisy_image
    '''
    assert len(fake_noise_map.shape) == len(real_noise_map.shape) == 3, 'The input should be 3D tensor'
    assert fake_noise_map.shape == real_noise_map.shape, 'The shape of fake_noisy_image, real_noisy_image and clean_image should be the same'
    assert fake_noise_map.min() >= -1 and fake_noise_map.max() <= 1, 'The value of fake_noisy_image should be in [-1, 1]'
    assert real_noise_map.min() >= -1 and real_noise_map.max() <= 1, 'The value of real_noisy_image should be in [-1, 1]'
    
    sigma_real = estimate_sigma_gauss_with_noise_map(real_noise_map)
    sigma_fake = estimate_sigma_gauss_with_noise_map(fake_noise_map)
    return kl_gauss_zero_center(sigma_fake, sigma_real).item()

if __name__ == '__main__':
    # Example
    torch.manual_seed(0)
    fake_noisy_image, real_noisy_image, clean_image = torch.rand(3, 3, 256, 256)
    akld_value = akld(fake_noisy_image, real_noisy_image, clean_image)
    akld_value_with_noise_map = akld_with_noise_map(fake_noisy_image - clean_image, real_noisy_image - clean_image)
    print(akld_value, akld_value_with_noise_map)
