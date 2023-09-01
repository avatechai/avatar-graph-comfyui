
class LoadBPY:
    def __init__(self):
        import bpy

        self.bpy = bpy
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # For disabling cache
            },
        }

    RETURN_TYPES = ("BPY",)
    RETURN_NAMES = ("bpy",)

    FUNCTION = "process"

    # OUTPUT_NODE = False

    CATEGORY = "mesh"

    def process(self, seed):
        self.bpy.ops.wm.read_factory_settings(use_empty=True)
        return (self.bpy,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadBPY": LoadBPY
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBPY": "Load BPY"
}
