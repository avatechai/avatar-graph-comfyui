import torch
from einops import rearrange, repeat
import numpy as np
import cv2

class SAM_Predict:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "embeddings": ("EMBEDDINGS",),
                "predictor": ("SAMPREDICTOR",),
                "prompt": ("SAM_PROMPT", )
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE", "IMAGE")
    RETURN_NAMES = ("image","out_image", "mask")

    FUNCTION = "segment"

    CATEGORY = "image"

    def segment(self, image, embeddings, predictor, prompt, mask=None):
        image_embedding_list = embeddings['image_embedding']
        shape = tuple(embeddings['shape'])
        input_size = tuple(embeddings['input_size'])

        # Convert the list back to a numpy array with the original shape
        image_embedding_np = np.array(image_embedding_list, dtype=np.single).reshape(shape)

        # Convert the numpy array to a PyTorch tensor
        image_embedding_tensor = torch.from_numpy(image_embedding_np)

        # Set the image embeddings for the model
        # predictor.set_torch_image(image_embedding_tensor, image.shape[:2])
        # print(image[0].shape[:2])
        predictor.input_size = input_size
        predictor.features = image_embedding_tensor
        predictor.is_image_set = True
        predictor.original_size = image[0].shape[:2]

        # prompt = [{"x":364,"y":153,"label":1},{"x":296,"y":189,"label":1},{"x":277,"y":246,"label":1}]

        # if point_1 != None:
            # x, y, z = point_1

        # point_coords = np.array([[x, y]])
        # point_labels = np.array([1])

        if prompt == None or len(prompt) == 0:
            return (image, image, None)

        point_coords = np.array([[p['x'], p['y']] for p in prompt])
        point_labels = np.array([p['label'] for p in prompt])

        masks, iou_predictions, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
        )

        if mask != None:
            # scale the mask to 256x256
            cv2_mask = cv2.resize(np.array(mask[0]), (256, 256))
            cv2_mask = cv2_mask[np.newaxis, :, :]
            cv2_mask = (cv2_mask * 255).astype(int)

            true_locations = np.array(np.where(cv2_mask[0] == 255))
            if true_locations.shape[1] > 0:
                # Randomly select a point in the mask
                rand_index = np.random.randint(true_locations.shape[1])
                y, x = true_locations[:, rand_index]
                point_coords = np.array([[x, y]])
                point_labels = np.array([1])

                masks, iou_predictions, low_res_masks = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=cv2_mask
                )
            else:
                # No detected mask
                h, w, c = image[0].shape
                masks = np.zeros((1, h, w))

        masks = torch.from_numpy(masks)

        masks = rearrange(masks[0], 'h w -> 1 h w')
        # masks = rearrange(masks, 'c h w -> 1 c h w')
        out_image = repeat(masks, '1 h w -> 1 h w c', c=3) * image

        print(masks.shape, torch.max(masks), torch.min(masks))
        print(image.shape, torch.max(image), torch.min(image))

        # print(emb)

        # print(masks, out_image)
        return (image, out_image, masks)

NODE_CLASS_MAPPINGS = {
    "SAM_Predict": SAM_Predict
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_Predict": "SAM Predict"
}
