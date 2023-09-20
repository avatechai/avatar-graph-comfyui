import blender_node
from mesh_utils import genreate_mesh_from_texture

class Object_MeshFromTexture(blender_node.ObjectOps):

    BASE_INPUT_TYPES = {}

    EXTRA_INPUT_TYPES = {
        "image": ("IMAGE",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
    }

    RETURN_TYPES = ("IMAGE", "BPY_OBJ")

    def blender_process(self, bpy, image, seed):
        return genreate_mesh_from_texture(bpy, image)