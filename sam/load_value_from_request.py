class LoadValueFromRequest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (
                    "STRING",
                    {"multiline": False, "default": "key_name"},
                ),
            },
            "optional": {
                "value": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)

    FUNCTION = "run"

    CATEGORY = "image"

    def run(self, name, value=None):
        if name:
            value = name
        return (value,)


NODE_CLASS_MAPPINGS = {"LoadValueFromRequest": LoadValueFromRequest}

NODE_DISPLAY_NAME_MAPPINGS = {"LoadValueFromRequest": "Load Value From Request"}
