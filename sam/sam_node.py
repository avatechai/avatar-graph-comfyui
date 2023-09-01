import folder_paths
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from einops import rearrange, repeat

class SAM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "point_1": ("VECTOR3D",),
                "model_type": (["vit_h", "vit_l", "vit_b"],),
                "ckpt": (folder_paths.get_filename_list("sam"),),
            },
            "optional": {
                "point_2": ("VECTOR3D",),
                "point_3": ("VECTOR3D",),
                "point_4": ("VECTOR3D",),
                "point_5": ("VECTOR3D",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask_1", "image_1", "mask_2", "image_2", "mask_3", "image_3", "mask_4", "image_4", "mask_5", "image_5",)

    FUNCTION = "segment"

    CATEGORY = "image"

    def is_point_in_bbox(self, point, mask):
        x, y, z = point
        in_bbox = mask[y, x].item() if y < mask.shape[0] and x < mask.shape[1] else None
        return in_bbox

    def find_largest_bbox(self, point, masks):
        largest_bbox = None
        largest_area = 0

        for mask in masks:
            area = mask['area']
            if self.is_point_in_bbox(point, mask['segmentation']) and area > largest_area:
                largest_bbox = mask
                largest_area = area

        return largest_bbox

    def segment(self, image, point_1, model_type, ckpt, point_2=None, point_3=None, point_4=None, point_5=None):
        image = (image[0] * 255).to(torch.uint8).numpy()
        H, W, C = image.shape
        ckpt = folder_paths.get_full_path("sam", ckpt)
        sam = sam_model_registry[model_type](checkpoint=ckpt) #.to("mps")
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        points = [point_1, point_2, point_3, point_4, point_5]

        outputs = []
        for point in points:
            if point is not None:
                bbox = self.find_largest_bbox(point, masks)
                mask = torch.from_numpy(bbox["segmentation"]) if bbox is not None else torch.zeros(H, W)
                mask = repeat(mask, 'h w -> h w c', c=3)
                out_image = torch.from_numpy(image) * mask

                mask = rearrange(mask, 'h w c -> 1 h w c')
                out_image = rearrange(out_image, 'h w c -> 1 h w c') / 255

                outputs.append(mask)
                outputs.append(out_image)
            else:
                outputs.append(None)
                outputs.append(None)
        return outputs

NODE_CLASS_MAPPINGS = {
    "Segmentation": SAM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Segmentation": "Segmentation Node"
}
