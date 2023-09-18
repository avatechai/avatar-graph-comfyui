import folder_paths
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from einops import rearrange, repeat
import os


class SAM_Prompt_Image:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {
            "required":
                {
                    "image": ('IMAGE', ),
                    # "image": (sorted(files), ),
                    "image_prompts_json": ("STRING", {
                        "multiline": False,
                        "default": "[]"
                    }),
                },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("SAM_PROMPT", )
    FUNCTION = "load_image"

    def load_image(self, image, image_prompts_json):
        import json

        image_prompts = json.loads(image_prompts_json)

        result = (image_prompts, )

        if isinstance(image_prompts, list):
            pass
        elif all(isinstance(item, list) for item in image_prompts.values()):
            for item in image_prompts.values():
                result.extend(item)

        return result

NODE_CLASS_MAPPINGS = {
    "SAM_Prompt_Image": SAM_Prompt_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Prompt_Image": "SAM_Prompt_Image "
}
