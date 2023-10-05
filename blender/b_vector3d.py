class VECTOR3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("FLOAT", {
                    "default": 0,
                    "step": 0.01,
                    "display": "number"
                }),
                "y": ("FLOAT", {
                    "default": 0,
                    "step": 0.01,
                    "display": "number"
                }),
                "z": ("FLOAT", {
                    "default": 0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("B_VECTOR3",)

    FUNCTION = "run"

    CATEGORY = "blender"

    def run(self, x, y, z):
        return ((x, y, z),)


NODE_CLASS_MAPPINGS = {
    "B_VECTOR3": VECTOR3D
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "B_VECTOR3": "Vector 3D (Blender)"
}
