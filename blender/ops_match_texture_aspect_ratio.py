import blender_node
from mesh_utils import genreate_mesh_from_texture


class Object_MatchTextureAspectRatio(blender_node.EditOps):

    EXTRA_INPUT_TYPES = {
        "image": ("IMAGE",),
        "scale": ('FLOAT', {'default': 0.001, })
    }

    CUSTOM_NAME = "Match Texture Aspect Ratio"

    def blender_process(self, bpy, BPY_OBJ, image, scale):
        height, width = image[0].shape[:2]
        width, height = (width * scale, height * scale, )

        bpy.ops.mesh.select_all(action='SELECT')
        # Transform resize
        bpy.ops.transform.resize(value=(width, height, 0))
        
