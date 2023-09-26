
class CreateShapeFlow():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape_flow": ("STRING", {
                    "multiline": False,
                    "default": "{node: ''}"
                }),
            },
        }

    RETURN_TYPES = ("shape_flow",)
    RETURN_NAMES = ("shape_flow",)
    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, blendshapes):
        return (blendshapes,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CreateShapeFlow": CreateShapeFlow
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateShapeFlow": "CreateShapeFlow"
}
