
class CreateShapeFlow():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape_flow": ("STRING", {
                    "multiline": False,
                    "default": '{"nodes": ""}'
                }),
            },
        }

    RETURN_TYPES = ("SHAPE_FLOW",)
    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, shape_flow):
        return (shape_flow,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CreateShapeFlow": CreateShapeFlow
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateShapeFlow": "Create Shape Flow"
}
