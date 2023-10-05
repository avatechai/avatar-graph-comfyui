import folder_paths
import os
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import json

class ImageAlphaMaskMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",) ,
                    "mask": ("MASK",) },
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image, mask):
        if image.shape[1] == mask.shape[0] and image.shape[2] == mask.shape[1]:
            image = torch.cat((image, 1 - mask.unsqueeze(0).unsqueeze(3)), dim=3)
        return (image, )

NODE_CLASS_MAPPINGS = {
     "Image Alpha Mask Merge": ImageAlphaMaskMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Alpha Mask Merge": "Image Alpha Mask Merge"
}