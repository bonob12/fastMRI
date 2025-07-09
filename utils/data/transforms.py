import numpy as np
import torch

from utils.model.fastmri import fft2c, ifft2c

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class FastmriDataTransform:
    def __init__(
        self, 
        isforward: bool = False,
        max_key: str = 'max',
    ):
        self.isforward = isforward
        self.max_key = max_key

    def _crop_and_pad_mask(self, mask):
        length = len(mask)
    
        if length > 384:
            start = (length - 384) // 2
            mask = mask[start : start + 384]
        elif length < 384:
            pad_len = 384 - length
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            mask = np.pad(mask, (pad_left, pad_right), mode='constant', constant_values=0)
            
        return mask.astype(np.float32)

    def _crop_if_needed(self, image):
        w_from = h_from = 0
        
        if 384 < image.shape[-3]:
            w_from = (image.shape[-3] - 384) // 2
            w_to = w_from + 384
        else:
            w_to = image.shape[-3]
        
        if 384 < image.shape[-2]:
            h_from = (image.shape[-2] - 384) // 2
            h_to = h_from + 384
        else:
            h_to = image.shape[-2]

        return image[..., w_from:w_to, h_from:h_to, :]
    
    def _pad_if_needed(self, image):
        pad_w = 384 - image.shape[-3]
        pad_h = 384 - image.shape[-2]
        
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
            
        if pad_h > 0:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h - pad_h_left
        else:
            pad_h_left = pad_h_right = 0 
            
        return torch.nn.functional.pad(image.permute(0, 3, 1, 2), (pad_h_left, pad_h_right, pad_w_left, pad_w_right), 'reflect').permute(0, 2, 3, 1)
        
    def _to_uniform_size(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        kspace = fft2c(image)
        return kspace
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        kspace = self._to_uniform_size(kspace)
        mask = self._crop_and_pad_mask(mask)
        mask = torch.from_numpy(mask.reshape(1, 1, 384, 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice