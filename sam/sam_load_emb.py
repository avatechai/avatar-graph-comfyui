import folder_paths
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from einops import rearrange, repeat

class SAM_Load_Embedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "embeddings"
                }),
            }
        }

    RETURN_TYPES = ("EMBEDDINGS",)
    RETURN_NAMES = ("EMBEDDINGS",)

    OUTPUT_NODE = True

    FUNCTION = "process"

    CATEGORY = "image"

    def process(self, filename):
        import json
        import numpy as np

        data = {}
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert list to numpy ndarray
        data['image_embedding'] = np.array(data['image_embedding'])
        
        return (data, )


NODE_CLASS_MAPPINGS = {
    "SAM_Load_Embedding": SAM_Load_Embedding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Load_Embedding": "SAM_Load_Embedding "
}
