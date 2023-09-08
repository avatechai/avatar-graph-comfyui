import blender_node


class GroupOps(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "BPY_OBJ2": (blender_node.BPY_OBJ,)
    }
    RETURN_TYPES = (blender_node.BPY_OBJS,)

    def blender_process(self, bpy, BPY_OBJ, BPY_OBJ2, **props):
        return ([BPY_OBJ, BPY_OBJ2],)

