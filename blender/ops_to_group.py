import blender_node


class ToGroupOps(blender_node.ObjectOps):
    RETURN_TYPES = (blender_node.BPY_OBJS,)

    def blender_process(self, bpy, BPY_OBJ, **props):
        return ([BPY_OBJ,],)