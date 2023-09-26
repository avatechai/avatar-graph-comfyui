
class CreateShapesFlow():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blendshapes": ("STRING", {
                    "multiline": False,
                    "default": "{node: ''}"
                }),
            },
        }

    RETURN_TYPES = ("blendshapes",)
    RETURN_NAMES = ("blendshapes",)
    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, blendshapes):
        return (blendshapes,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CreateShapesFlow": CreateShapesFlow
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateShapesFlow": "CreateShapesFlow"
}
