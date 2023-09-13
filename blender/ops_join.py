import blender_node


class Mesh_JoinMesh(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "BPY_OBJ2": (blender_node.BPY_OBJ,)
    }

    def blender_process(self, bpy, BPY_OBJ, BPY_OBJ2, **props):

        override = bpy.context.copy()
        override["active_object"] = BPY_OBJ
        override["selected_editable_objects"] = list([BPY_OBJ, BPY_OBJ2])
        bpy.ops.object.join(override)

        return (BPY_OBJ,)

