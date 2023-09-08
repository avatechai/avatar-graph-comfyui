import blender_node


class MeshSelectAllOps(blender_node.EditOps):
    EXTRA_INPUT_TYPES = {
        "action": (['SELECT', 'TOGGLE', 'DESELECT', 'INVERT'],),
    }

    def blender_process(self, bpy, BPY_OBJ, **props):
        bpy.ops.mesh.select_all(**props)
