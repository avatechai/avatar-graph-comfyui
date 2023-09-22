import folder_paths
import os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from einops import rearrange, repeat


global_predictor = None

class SAM:
    def __init__(self):
        self.predictor = None
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
                "model_type": (["vit_h", "vit_l", "vit_b"],),
                "ckpt": (folder_paths.get_filename_list("sam"),),
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

    def load_image(self, image, model_type, ckpt, embedding_id, image_prompts_json):
        import json

        global global_predictor

        if global_predictor is None:
            ckpt = folder_paths.get_full_path("sam", ckpt)
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            predictor = SamPredictor(sam)
            global_predictor = predictor
        
        predictor = global_predictor
        
        emb_filename = f"{self.output_dir}/{embedding_id}.npy"
        if not os.path.exists(emb_filename):
            image_np = (image[0].numpy() * 255).astype(np.uint8)
            predictor.set_image(image_np)
            emb = predictor.get_image_embedding().cpu().numpy()
            np.save(emb_filename, emb)
        else:
            emb = np.load(emb_filename)

            with open(f"{self.output_dir}/{embedding_id}.json") as f:
                data = json.load(f)
                predictor.input_size = data["input_size"]
                predictor.features = torch.from_numpy(emb)
                predictor.is_image_set = True
                predictor.original_size = data["original_size"]

        image_prompts = json.loads(image_prompts_json)

        result = [image_prompts]

        if isinstance(image_prompts, list):
            pass
        elif all(isinstance(item, list) for item in image_prompts.values()):
            for item in image_prompts.values():
                if (len(item) == 0):
                    h, w, c = image[0].shape
                    result.append(torch.zeros(1, h, w, c))
                    continue
                point_coords = np.array([[p['x'], p['y']] for p in item])
                point_labels = np.array([p['label'] for p in item])

                masks, _, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
                masks = torch.from_numpy(masks)
                masks = rearrange(masks[0], 'h w -> 1 h w')
                out_image = repeat(masks, '1 h w -> 1 h w c', c=3) * image
                result.append(out_image)
        return result


NODE_CLASS_MAPPINGS = {"SAM": SAM}

NODE_DISPLAY_NAME_MAPPINGS = {"SAM": "SAM"}
