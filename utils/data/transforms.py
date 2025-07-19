import numpy as np
import torch
import torchvision.transforms.functional as TF
from utils.model.fastmri import fft2c, ifft2c, rss, complex_abs
from typing import Sequence

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data.astype(np.float32))

class CustomMaskFunc():
    def __init__(self, acceleration: int):
        self.acceleration = acceleration

    def __call__(self, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * 0.08))

        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        offset = (num_cols // 2) // self.acceleration

        accel_samples = np.arange(offset, num_cols - 1, self.acceleration)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True
        mask = to_tensor(mask.reshape(1, 1, num_cols, 1))

        return mask

class FastmriDataTransform:
    def __init__(
        self, 
        data_type: str = 'train',
        max_key: str = 'max',
        acceleration: int = 4,
        augmentor = None,
    ):
        self.data_type = data_type
        self.max_key = max_key
        if self.data_type == 'train':
            self.mask_func = CustomMaskFunc(acceleration)
            self.augmentor = augmentor
    
    def update_epoch(self, epoch):
        self.augmentor.update_epoch(epoch)
    
    def __call__(self, mask, input, target, attrs, fname, slice_idx):
        if self.data_type != 'test':
            target = to_tensor(target)
            maximum = attrs[self.max_key]

        kspace = to_tensor(input)
        if self.data_type == 'train':
            kspace, target, maximum = self.augmentor(kspace, target, maximum)
            mask = self.mask_func(kspace.shape)
            kspace = kspace * mask + 0.0
        else:
            if 384 < kspace.shape[-3]:
                image = ifft2c(kspace)
                h_from = (image.shape[-3] - 384) // 2
                h_to = h_from + 384
                image = image[..., h_from:h_to, :, :]
                kspace = fft2c(image)
            if self.data_type == 'test':
                return kspace
            mask = to_tensor(mask.reshape(1, 1, kspace.shape[-2], 1))
        
        mask = mask.to(torch.uint8)
        return mask, kspace, target, maximum, fname, slice_idx