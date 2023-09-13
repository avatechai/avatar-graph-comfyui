class B_ENUM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enum": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("B_ENUM",)

    FUNCTION = "run"

    CATEGORY = "blender"

    def run(self, x, y, z, u):
        return ((x, y, z, u),)


NODE_CLASS_MAPPINGS = {
    "B_ENUM": B_ENUM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "B_ENUM": "ENUM (Blender)"
}
