import torch
from einops import rearrange
from nodes import PreviewImage

class PointVisualizer(PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "point": ("VECTOR3D",),
                "point_size": ("INT", {
                    "default": 1, 
                    "min": 10,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    FUNCTION = "process"

    CATEGORY = "image"

    def process(self, images, point, point_size):
        x, y, z = point
        h, w, c = images[0].shape
        point_image = torch.zeros(images[0].shape)

        center_x, center_y, _ = point
        top_left_x = center_x - point_size // 2
        top_left_y = center_y - point_size // 2

        # Make sure the square is within the image boundaries
        top_left_x = max(0, min(w - point_size, top_left_x))
        top_left_y = max(0, min(h - point_size, top_left_y))

        point_image[top_left_y:top_left_y+point_size, top_left_x:top_left_x+point_size] = 1
        point_image = rearrange(point_image, 'h w c -> 1 h w c')

        # Blending
        images = images * 0.2 + point_image * 0.8
        return self.save_images(images)

NODE_CLASS_MAPPINGS = {
    "PointVisualizer": PointVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PointVisualizer": "Point Visualizer"
}
