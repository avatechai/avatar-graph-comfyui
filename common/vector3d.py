class VECTOR3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("FLOAT", {
                    "default": 0, 
                    "min": -1024,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "y": ("FLOAT", {
                    "default": 0, 
                    "min": -1024,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "z": ("FLOAT", {
                    "default": 0, 
                    "min": -1024,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("VECTOR3D",)

    FUNCTION = "run"

    CATEGORY = "utils"

    def run(self, x, y, z):
        return ([x, y, z],)

NODE_CLASS_MAPPINGS = {
    "VECTOR3D": VECTOR3D
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VECTOR3D": "Vector 3D"
}
