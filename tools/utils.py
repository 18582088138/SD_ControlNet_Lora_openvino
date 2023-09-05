import torch
import numpy as np
from typing import Union, List, Optional, Tuple


def scale_fit_to_window(dst_width:int, dst_height:int, image_width:int, image_height:int):
    """
    Preprocessing helper function for calculating image size for resize with peserving original aspect ratio 
    and fitting image to specific window size
    
    Parameters:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height
    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize
    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)

def randn_tensor_np(
    shape: Union[Tuple, List],
    dtype: Optional[np.dtype] = np.float32,
    ):
    """
    Helper function for generation random values tensor with given shape and data type
    
    Parameters:
      shape (Union[Tuple, List]): shape for filling random values
      dtype (np.dtype, *optiona*, np.float32): data type for result
    Returns:
      latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
    """
    latents = np.random.randn(*shape).astype(dtype)
    return latents

def randn_tensor_torch(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    ):
    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=torch.device("cpu") ,dtype=dtype, layout=None)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0)
    else:
        latents = torch.randn(shape, generator=generator, device=torch.device("cpu"), dtype=dtype, layout=None)
    latents = latents.numpy()
    return latents