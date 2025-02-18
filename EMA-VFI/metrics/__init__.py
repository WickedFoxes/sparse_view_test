from .psnr import psnr
from .ssim import ssim
from .akld import akld
from .kld import kld
from ._batch_wise import batch_wise

metrics = ['psnr', 'ssim', 'akld', 'kld']
__all__ = metrics.copy()
for func_name in metrics:
    func = locals()[func_name]
    batch_func_name = f"batch_{func_name}"
    locals()[batch_func_name] = batch_wise(func)
    __all__.append(batch_func_name) 
