import blender_node
from mesh_utils import assign_texture

class Object_AssignTexture(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "texture": ("IMAGE",),
        "texture_name": ("STRING", {
            "multiline": False,
            "default": "my_image"
        }),
    }

    def blender_process(self, bpy, BPY_OBJ, texture, texture_name):
        assign_texture(bpy, BPY_OBJ, texture, texture_name)