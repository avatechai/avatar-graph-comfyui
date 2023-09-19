import requests
from PIL import Image
import io
import numpy as np
import folder_paths

class SAM_Embedding:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "embedding_id": ("STRING", {
                    "multiline": False,
                    "default": "embedding"
                }),
            },
            "optional": {
                "predictor": ("SAMPREDICTOR",)
            }
        }

    RETURN_TYPES = ("EMBEDDINGS", )
    RETURN_NAMES = ("embeddings", )

    FUNCTION = "segment"

    CATEGORY = "image"

    def segment(self, image, embedding_id, predictor=None):
        # Convert PyTorch tensor to numpy array
        image_np = (image[0].numpy() * 255).astype(np.uint8)
        
        if predictor != None:
            predictor.set_image(image_np)
            emb = predictor.get_image_embedding().cpu().numpy()
            output = {
                "image_embedding": emb,
                "shape": emb.shape,
                "input_size": predictor.input_size
            }
        else:
            # Convert numpy array to PIL Image
            img = Image.fromarray(image_np)

            # Create an in-memory bytes buffer
            img_byte_arr = io.BytesIO()

            # Save the PIL Image to the bytes buffer in PNG format
            img.save(img_byte_arr, format='PNG')

            # Get the bytes value of the buffer
            img_byte_arr = img_byte_arr.getvalue()

            # Create a dictionary with the image bytes
            files = {'image': ('image.png', img_byte_arr)}

            # Send the POST request
            response = requests.post('https://avatechgg--segment-anything-entrypoint.modal.run', files=files)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                output = response.json()
            else:
                print(f"Request failed with status code {response.status_code}")
                output = None

            # sam = sam_model_registry[model_type](checkpoint=ckpt)
            # predictor = SamPredictor(sam)
            # masks = predictor.set_torch_image
            # masks = predictor.predict

            # print(output)

        np.save(f"{self.output_dir}/{embedding_id}.npy", output["image_embedding"])
        return (output, )

NODE_CLASS_MAPPINGS = {
    "SAM_Embedding": SAM_Embedding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Embedding": "SAM Embedding"
}
