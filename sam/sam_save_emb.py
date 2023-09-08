import folder_paths
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from einops import rearrange, repeat

class SAM_Save_Embedding:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeddings": ("EMBEDDINGS",),
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "embeddings"
                }),
                "write_mode": (["Overwrite", "Increment"],),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    FUNCTION = "process"

    CATEGORY = "image"

    def process(self, embeddings, filename, write_mode):
        import json
        import numpy as np

        filepath = self.output_dir + "/" + filename + '.json'

        if write_mode == "Increment":
            count = 0
            # while file exists, increment count
            while os.path.exists(self.output_dir + "/" + filename + '_' + str(count) + '.json'):
                count += 1

            filepath = self.output_dir + "/" + filename + '_' + str(count) + '.json'

        # print(embeddings)
        if isinstance(embeddings['image_embedding'], np.ndarray):
            embeddings['image_embedding'] = embeddings['image_embedding'].tolist()
        with open(filepath, 'w') as f:
            json.dump(embeddings, f)
        
        return { "ui" : { "file": { filepath.replace(f"{self.output_dir}/", "") } } }


NODE_CLASS_MAPPINGS = {
    "SAM_Save_Embedding": SAM_Save_Embedding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Save_Embedding": "SAM_Save_Embedding "
}
