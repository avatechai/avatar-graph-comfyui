import folder_paths
from PIL import Image, ImageOps
import numpy as np
import torch

class LoadImageFromRequest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (
                    "STRING",
                    {"multiline": False, "default": "face.png"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "image"

    def run(self, name, image=None):
        try:
            image_path = folder_paths.get_annotated_filepath(name)
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            # image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            return [image]
        except:
            return [image]


NODE_CLASS_MAPPINGS = {"LoadImageFromRequest": LoadImageFromRequest}

NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageFromRequest": "Load Image From Request"}
