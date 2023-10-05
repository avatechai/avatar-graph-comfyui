class VECTOR4D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("FLOAT", {
                    "default": 0,
                    "step":  0.01,
                    "display": "number"
                }),
                "y": ("FLOAT", {
                    "default": 0,
                    "step":  0.01,
                    "display": "number"
                }),
                "z": ("FLOAT", {
                    "default": 0,
                    "step":  0.01,
                    "display": "number"
                }),
                "u": ("FLOAT", {
                    "default": 0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("B_VECTOR4",)

    FUNCTION = "run"

    CATEGORY = "blender"

    def run(self, x, y, z, u):
        return ((x, y, z, u),)


NODE_CLASS_MAPPINGS = {
    "B_VECTOR4": VECTOR4D
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "B_VECTOR4": "Vector 4D (Blender)"
}
