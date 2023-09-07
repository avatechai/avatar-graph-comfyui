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
                    "image": (sorted(files), ),
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

        print(image_prompts)

        return (image_prompts, )

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {
    "SAM_Prompt_Image": SAM_Prompt_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Prompt_Image": "SAM_Prompt_Image "
}
