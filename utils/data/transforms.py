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

        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask
    

class Aug_HorizontalFlip:
    def __init__(self, p_flip: float = 0.5):
        self.p_flip = p_flip

    def __call__(self, x: torch.Tensor, p_aug: float):
        if torch.rand(1).item() < self.p_flip:
            return torch.flip(x, dims=[-2]), True
        return x, False


class Aug_RotateShiftShearScale:
    def __init__(self, p_rotate, p_shift, p_scale, p_shear, 
                 max_degree=5, max_shift_x=5, max_shift_y=5, scale_range=0.1, max_shear_x=5, max_shear_y=5):
        self.p_rotate = p_rotate
        self.p_shift = p_shift
        self.p_scale = p_scale
        self.p_shear = p_shear
        
        self.max_degree = max_degree
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.scale_range = scale_range
        self.max_shear_x = max_shear_x
        self.max_shear_y = max_shear_y

    def __call__(self, x: torch.Tensor, p_aug: float):
        angle = 0.0
        shift_x = shift_y = 0
        shear_x = shear_y = 0
        scale = 1.0
        auged = False
        if torch.rand(1).item() < self.p_rotate * p_aug:
            angle = float(torch.empty(1).uniform_(-self.max_degree, self.max_degree))
            auged = True
        if torch.rand(1).item() < self.p_shift * p_aug:
            shift_x = int(torch.empty(1).uniform_(-self.max_shift_x, self.max_shift_x + 1).item())
            shift_y = int(torch.empty(1).uniform_(-self.max_shift_y, self.max_shift_y + 1).item())
            auged = True
        if torch.rand(1).item() < self.p_shear * p_aug:
            shear_x = int(torch.empty(1).uniform_(-self.max_shear_x, self.max_shear_x + 1).item())
            shear_y = int(torch.empty(1).uniform_(-self.max_shear_y, self.max_shear_y + 1).item())
            auged = True
        if torch.rand(1).item() < self.p_scale * p_aug:
            scale = 1.0 + float(torch.empty(1).uniform_(-self.scale_range, self.scale_range))
            auged = True
        
        if auged:
            real = TF.affine(x[..., 0], angle=angle, translate=[shift_x, shift_y], scale=scale, shear=[shear_x, shear_y])
            imag = TF.affine(x[..., 1], angle=angle, translate=[shift_x, shift_y], scale=scale, shear=[shear_x, shear_y])
            return torch.stack((real, imag), axis=-1), True
        else:
            return x, False


# class Aug_BiasField:
#     def __init__(self, p_bias=0.2, strength=0.3):
#         self.p_bias = p_bias
#         self.strength = strength

#     def __call__(self, x: torch.Tensor, p_aug: float):
#         if torch.rand(1).item() < self.p_bias * p_aug:
#             strength = float(torch.empty(1).uniform_(-self.strength, self.strength))
#             bias = torch.tensor(
#                 np.outer(
#                     np.linspace(1 - strength, 1 + strength, x.shape[-3]),
#                     np.linspace(1 - strength, 1 + strength, x.shape[-2])
#                 ),
#                 dtype=x.dtype,
#                 device=x.device,
#             )
#             x = x * bias[..., None]
#             return x, True
#         return x, False


# class Aug_MotionBlur:
#     def __init__(self, p_blur=0.2, kernel_size=3):
#         self.p_blur = p_blur
#         self.kernel_size = kernel_size
    
#     def __call__(self, x: torch.Tensor, p_aug: float):
#         if torch.rand(1).item() < self.p_blur * p_aug:
#             real = TF.gaussian_blur(x[..., 0], kernel_size=self.kernel_size)
#             imag = TF.gaussian_blur(x[..., 1], kernel_size=self.kernel_size)
#             return torch.stack((real, imag), dim=-1), True
#         return x, False


class Aug_Contrast:
    def __init__(self, p_contrast, 
                 contrast_range=0.1):
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range

    def __call__(self, x: torch.Tensor, p_aug: float):
        contrast_factor = 1.0
        if torch.rand(1).item() < self.p_contrast * p_aug:
            contrast_factor = 1.0 + float(torch.empty(1).uniform_(-self.contrast_range, self.contrast_range))
            x=torch.view_as_complex(x)
            mag = torch.abs(x)
            phase = torch.angle(x)
            mag_mean = mag.mean(dim=(-2, -1), keepdim=True)

            mag = (mag - mag_mean) * contrast_factor + mag_mean
            mag = torch.clamp(mag, min=0.0)

            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)

            return torch.stack((real, imag), axis=-1), True
        else:
            return x, False
    

class FastmriDataTransform:
    def __init__(
        self, 
        data_type: str = 'train',
        max_key: str = 'max',
        aug_start_epoch: int = 0,
        aug_gamma: float = 0.1,
        acceleration: int = 4,
        task: str = 'brain',
    ):
        self.data_type = data_type
        self.max_key = max_key
        self.epoch = 0
        self.aug_start_epoch = aug_start_epoch
        self.aug_gamma = aug_gamma
        self.mask_func = CustomMaskFunc(acceleration)

        if task == 'brain':
            self.augmentations = [
                Aug_RotateShiftShearScale(
                    p_rotate=0.25, p_shift=0.5, p_scale=0.5, p_shear=0.5, 
                    max_degree=5, max_shift_x=5, max_shift_y=5, scale_range=0.1, max_shear_x=5, max_shear_y=5
                ),
                Aug_Contrast(p_contrast=0.25, contrast_range=0.1),
            ]
        elif task == 'knee':
            self.augmentations = [
                Aug_HorizontalFlip(p_flip=0.5),
                Aug_RotateShiftShearScale(
                    p_rotate=0.25, p_shift=0.5, p_scale=0.5, p_shear=0.5, 
                    max_degree=5, max_shift_x=5, max_shift_y=5, scale_range=0.1, max_shear_x=5, max_shear_y=5
                ),
                Aug_Contrast(p_contrast=0.25, contrast_range=0.1),
            ]
    
    def update_epoch(self, epoch):
        self.epoch = epoch

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
        
    def _to_uniform_size_image(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        return image
    
    def __call__(self, mask, input, target, attrs, fname, slice_idx):
        if self.data_type != 'test':
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = None
            maximum = None

        kspace = to_tensor(input)
        if self.data_type == 'train':
            image = self._to_uniform_size_image(kspace)
            p_aug = 1 - np.exp(-self.aug_gamma * max(0, self.epoch - self.aug_start_epoch))
            is_aug = False
            for aug in self.augmentations:
                image, auged = aug(image, p_aug)
                if auged: is_aug = True
            if is_aug:
                target = rss(complex_abs(image), dim=1)
            kspace = fft2c(image)
            mask = self.mask_func(kspace.shape)
            kspace = kspace * mask + 0.0
        else:
            if 384 < kspace.shape[-3]:
                image = ifft2c(kspace)
                h_from = (image.shape[-3] - 384) // 2
                h_to = h_from + 384
                image = image[..., h_from:h_to, :, :]
                kspace = fft2c(image)
            mask_shape = [1 for _ in kspace.shape]
            mask_shape[-2] = kspace.shape[-2]
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        
        mask = mask.to(torch.uint8)
        return mask, kspace, target, maximum, fname, slice_idx