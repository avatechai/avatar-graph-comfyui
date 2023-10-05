import folder_paths
import os
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import json

class SaveImageWithWorkflow:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": 
                   {"image": (sorted(files), {"image_upload": True}),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"})},
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
               }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, image, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        images=(image)
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            metadata = PngInfo()
            if prompt is not None:
                # prompt = json.loads(prompt)
                prompt = {k: v for k, v in prompt.items() if v['class_type'] != 'Save Image With Workflow'}
                metadata.add_text("prompt", json.dumps(prompt))

            print(extra_pnginfo)

            # if prompt is not None:
            #     metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    if (x == 'workflow'):
                        extra_pnginfo[x]["nodes"] = [node for node in extra_pnginfo[x]["nodes"] if node['type'] != 'Save Image With Workflow']

                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
     "Save Image With Workflow": SaveImageWithWorkflow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image With Workflow": "Save Image With Workflow"
}