import blender_node
from mesh_utils import genreate_mesh_from_texture, assign_texture

class Object_UV_Modifier(blender_node.EditOps):

    EXTRA_INPUT_TYPES = {
        "scale": ('FLOAT', {'default': 1, "display": "number", "step": 0.01}),
        "texture_name": ('STRING', {'default': 'Texture', })
    }

    CUSTOM_NAME = "UV Modifier"

    def blender_process(self, bpy, BPY_OBJ, scale, texture_name):
        import bmesh

        bm = bmesh.from_edit_mesh(BPY_OBJ.data)

        uv_layer = bm.loops.layers.uv.verify()
        for f in bm.faces:
            # move all of the UVs in this face up one UDIM tile
            for l in f.loops:
                l[uv_layer].uv = (l[uv_layer].uv[0], 0.998 if l[uv_layer].uv[1] == 1 else l[uv_layer].uv[1])

        bmesh.update_edit_mesh(BPY_OBJ.data)