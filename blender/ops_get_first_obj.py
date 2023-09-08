import blender_node


class GetFirstObjOps(blender_node.ObjectOps):
    BASE_INPUT_TYPES = {
        **blender_node.BPY_OBJS_TYPE
    }

    def blender_process(self, bpy, BPY_OBJS, **props):
        return (BPY_OBJS[0], )