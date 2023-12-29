
class CombinePoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
        }

    RETURN_TYPES = ("SAM_PROMPTS",)

    FUNCTION = "run"

    CATEGORY = "image"

    # OUTPUT_NODE = True

    def run(self, *args, **kwargs):
        print("args", args)
        print("kwargs", kwargs)
        return ([])


NODE_CLASS_MAPPINGS = {"Combine Points": CombinePoints}

NODE_DISPLAY_NAME_MAPPINGS = {"Combine Points": "Combine Points"}
