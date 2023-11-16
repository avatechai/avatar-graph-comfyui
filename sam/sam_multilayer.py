import folder_paths
import os
import numpy as np
import torch
import re
import json
from segment_anything import sam_model_registry, SamPredictor
from einops import rearrange, repeat
from PIL import Image


global_predictor = None

class SAMMultiLayer:
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
        image_prompts = json.loads(image_prompts_json)
        
        order_file = f"{self.output_dir}/segments_{embedding_id}/order.json"
        if os.path.exists(order_file):
            # Frontend uploads segments images to backend => backend reads all segments images and passes them to next nodes
            with open(order_file) as f:
                order = json.load(f)

            result = [image_prompts]

            for segment in order:
                image = Image.open(f"{self.output_dir}/segments_{embedding_id}/{segment}.png")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                result.append(image)

            return result
        else:
            # Frontend uploads clicks coordinates to backend => backend runs SAM and passes the segments to next nodes
            global global_predictor
            model_type = re.findall(r'vit_[lbh]', ckpt)[0]

            if global_predictor is None:
                ckpt = folder_paths.get_full_path("sams", ckpt)
                sam = sam_model_registry[model_type](checkpoint=ckpt)
                predictor = SamPredictor(sam)
                global_predictor = predictor
            
            predictor = global_predictor

            if image.shape[3] == 4:
                image = image[:, :, :, :3]

            emb_filename = f"{self.output_dir}/{embedding_id}_{model_type}.npy"
            if not os.path.exists(emb_filename):
                image_np = (image[0].numpy() * 255).astype(np.uint8)
                predictor.set_image(image_np)
                emb = predictor.get_image_embedding().cpu().numpy()
                np.save(emb_filename, emb)

                with open(f"{self.output_dir}/{embedding_id}_{model_type}.json", "w") as f:
                    data = {
                        "input_size": predictor.input_size,
                        "original_size": predictor.original_size,
                    }
                    json.dump(data, f)
            else:
                emb = np.load(emb_filename)

                with open(f"{self.output_dir}/{embedding_id}_{model_type}.json") as f:
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


NODE_CLASS_MAPPINGS = {"SAM MultiLayer": SAMMultiLayer}

NODE_DISPLAY_NAME_MAPPINGS = {"SAM MultiLayer": "SAM MultiLayer"}
