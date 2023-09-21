import blender_node
from mesh_utils import genreate_mesh_from_texture

class Object_AddShapeKeys(blender_node.ObjectOps):

    EXTRA_INPUT_TYPES = {
        "shape_keys": ("STRING", {"default": "key1,key2", "multiline": True}),
        "from_mix": ("BOOLEAN", {"default": False})
    }

    def blender_process(self, bpy, BPY_OBJ, shape_keys, from_mix):
        shape_key_names = shape_keys.split(',')
        for name in shape_key_names:
            BPY_OBJ.shape_key_add(name=name, from_mix=from_mix)
        
