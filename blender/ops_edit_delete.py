import blender_node


class EditDeleteOps(blender_node.EditOps):
    EXTRA_INPUT_TYPES = {
        "type": (['VERT', 'EDGE', 'FACE', 'EDGE_FACE', 'ONLY_FACE'],),
    }

    def blender_process(self, bpy, BPY_OBJ, **props):
        bpy.ops.mesh.delete(**props)