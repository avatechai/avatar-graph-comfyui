import torch
from einops import rearrange, repeat
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
import requests
import json
import io

class AvatarSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("back_hair", "front_hair","eyes", "eyelashes", "mouth")

    FUNCTION = "segment"

    CATEGORY = "image"

    def segment(self, image):
        # Convert tensor to PIL image
        image = image[0]
        image = rearrange(image, 'h w c -> c h w')
        image = transforms.ToPILImage()(image)
        
        # Save image to in-memory file
        image_buff = io.BytesIO()
        image.save(image_buff, format="PNG")
        image_string = base64.b64encode(image_buff.getvalue()).decode("utf-8") 

        # Send request to API
        url = 'https://q41iq6s6t8.execute-api.ap-southeast-1.amazonaws.com/seg-cpu'
        headers = {
            'content-type': 'application/json'
        }
        data = {
            'img_str': image_string
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Convert response to tensors
        outs = []
        masks = response.json()
        for segment_name, img_str in masks.items():
            mask = base64.b64decode(img_str.encode('utf-8'))
            mask = Image.open(io.BytesIO(mask))
            mask = np.array(mask)
            mask = torch.from_numpy(mask) # shape: H, W, 3
            mask = rearrange(mask, 'h w c -> 1 h w c')
            outs.append(mask)
        return outs

NODE_CLASS_MAPPINGS = {
    "AvatarSegmentation": AvatarSegmentation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AvatarSegmentation": "Avatar Segmentation"
}
