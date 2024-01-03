import json

class CombinePoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
        }

    RETURN_NAMES = ("SAM_PROMPTS",)
    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "image"

    # OUTPUT_NODE = True

    def run(self, *args, **kwargs):
        sam_prompts = json.dumps(kwargs, default=str)
        return (sam_prompts,)


NODE_CLASS_MAPPINGS = {"Combine Points": CombinePoints}

NODE_DISPLAY_NAME_MAPPINGS = {"Combine Points": "Combine Points"}
