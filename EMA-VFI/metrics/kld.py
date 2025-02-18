import numpy as np
import torch

def get_histogram(data, bin_edges=None, left_edge=-1.0, right_edge=1.0, n_bins=256):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers

def _kld(p_data: torch.Tensor, q_data: torch.Tensor, bin_edges=None, left_edge=-1.0, right_edge=1.0, n_bins=256):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    assert p_data.shape == q_data.shape
    assert len(p_data.shape) == 3
    p_data = p_data.cpu()
    q_data = q_data.cpu()

    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    return kl_fwd

def kld(fake_noisy_image: torch.Tensor, real_noisy_image: torch.Tensor, clean_image: torch.Tensor):
    '''
    Calculate the KLD between fake_noisy_image and real_noisy_image
    '''
    assert len(fake_noisy_image.shape) == len(real_noisy_image.shape) == len(clean_image.shape) == 3, 'The input should be 3D tensor'
    assert fake_noisy_image.shape == real_noisy_image.shape == clean_image.shape, 'The shape of fake_noisy_image, real_noisy_image and clean_image should be the same'
    assert fake_noisy_image.min() >= 0 and fake_noisy_image.max() <= 1, 'The value of fake_noisy_image should be in [0, 1]'
    assert real_noisy_image.min() >= 0 and real_noisy_image.max() <= 1, 'The value of real_noisy_image should be in [0, 1]'
    assert clean_image.min() >= 0 and clean_image.max() <= 1, 'The value of clean_image should be in [0, 1]'
    
    real_noise_map = real_noisy_image - clean_image
    fake_noise_map = fake_noisy_image - clean_image
    return _kld(real_noise_map, fake_noise_map)

def kld_with_noise_map(fake_noise_map: torch.Tensor, real_noise_map: torch.Tensor):
    '''
    Calculate the KLD between fake_noisy_image and real_noisy_image
    '''
    assert len(fake_noise_map.shape) == len(real_noise_map.shape) == 3, 'The input should be 3D tensor'
    assert fake_noise_map.shape == real_noise_map.shape, 'The shape of fake_noisy_image, real_noisy_image and clean_image should be the same'
    assert fake_noise_map.min() >= -1 and fake_noise_map.max() <= 1, 'The value of fake_noisy_image should be in [-1, 1]'
    assert real_noise_map.min() >= -1 and real_noise_map.max() <= 1, 'The value of real_noisy_image should be in [-1, 1]'
    return _kld(real_noise_map, fake_noise_map)

if __name__ == '__main__':
    # Example
    torch.manual_seed(0)
    fake_noisy_image, real_noisy_image, clean_image = torch.rand(3, 3, 256, 256)
    print(kld(fake_noisy_image, real_noisy_image, clean_image))
    print(kld_with_noise_map(fake_noisy_image - clean_image, real_noisy_image - clean_image))