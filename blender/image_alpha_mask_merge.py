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
        image = torch.cat((image, 1 - mask.unsqueeze(0).unsqueeze(3)), dim=3)
        return (image, )

NODE_CLASS_MAPPINGS = {
     "ImageAlphaMaskMerge": ImageAlphaMaskMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAlphaMaskMerge": "ImageAlphaMaskMerge"
}