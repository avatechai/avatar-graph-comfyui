import blender_node
from mesh_utils import genreate_mesh_from_texture, assign_texture

class Object_PlaneTextureUnwrap(blender_node.EditOps):

    EXTRA_INPUT_TYPES = {
        "image": ("IMAGE",),
        "scale": ('FLOAT', {'default': 1, }),
        "texture_name": ('STRING', {'default': 'Texture', })
    }

    def blender_process(self, bpy, BPY_OBJ, image, scale, texture_name):
        bpy.ops.mesh.select_all(action='SELECT')
        # Mesh fill
        bpy.ops.mesh.fill(
            use_beauty=True,
        )
        # Cube project
        bpy.ops.uv.cube_project(
            cube_size=scale,
            scale_to_bounds=True,
        )

        assign_texture(bpy, BPY_OBJ, image, texture_name)



        

        
        
