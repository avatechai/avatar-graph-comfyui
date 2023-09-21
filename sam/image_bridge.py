class ImageBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "image"

    def run(self, image):
        return [image]

NODE_CLASS_MAPPINGS = {
    "ImageBridge": ImageBridge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBridge": "Image Bridge"
}
