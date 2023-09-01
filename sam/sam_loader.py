import folder_paths
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from einops import rearrange, repeat
import requests
from PIL import Image
import io
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from einops import rearrange, repeat

class SAM_Remote_Emb:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["vit_h", "vit_l", "vit_b"],),
                "ckpt": (folder_paths.get_filename_list("sam"),),
            },
        }

    RETURN_TYPES = ("SAM", "SAMPREDICTOR")
    RETURN_NAMES = ("sam", "predictor")

    FUNCTION = "segment"

    CATEGORY = "image"

    def segment(self, model_type, ckpt):
        ckpt = folder_paths.get_full_path("sam", ckpt)
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        predictor = SamPredictor(sam)

        return (sam, predictor)

NODE_CLASS_MAPPINGS = {
    "SAM_Loader": SAM_Remote_Emb
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Loader": "SAM Loader"
}
