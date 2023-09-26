import folder_paths
import os
import numpy as np
import re
from segment_anything import sam_model_registry, SamPredictor


class SAM_Prompt_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "ckpt": (folder_paths.get_filename_list("sams"),),
                "embedding_id": (
                    "STRING",
                    {"multiline": False, "default": "embedding"},
                ),
                # "image": (sorted(files), ),
                "image_prompts_json": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("SAM_PROMPT",)
    FUNCTION = "load_image"

    def load_image(self, image, ckpt, embedding_id, image_prompts_json):
        import json

        emb_filename = f"{self.output_dir}/{embedding_id}.npy"
        if not os.path.exists(emb_filename):
            ckpt = folder_paths.get_full_path("sams", ckpt)
            model_type = re.findall(r'vit_[lbh]', ckpt)[0]
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            predictor = SamPredictor(sam)

            image_np = (image[0].numpy() * 255).astype(np.uint8)
            predictor.set_image(image_np)
            emb = predictor.get_image_embedding().cpu().numpy()
            np.save(emb_filename, emb)

        image_prompts = json.loads(image_prompts_json)

        result = (image_prompts,)

        if isinstance(image_prompts, list):
            pass
        elif all(isinstance(item, list) for item in image_prompts.values()):
            for item in image_prompts.values():
                result += (item,)

        return result


NODE_CLASS_MAPPINGS = {"SAM_Prompt_Image": SAM_Prompt_Image}

NODE_DISPLAY_NAME_MAPPINGS = {"SAM_Prompt_Image": "SAM_Prompt_Image "}
