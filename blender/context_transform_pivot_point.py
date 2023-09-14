import blender_node


class ContextSet_TransformPivotPoint(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "pivot": (['BOUNDING_BOX_CENTER', 'CURSOR', 'INDIVIDUAL_ORIGINS', 'MEDIAN_POINT', 'ACTIVE_ELEMENT'],)
    }

    def blender_process(self, bpy, BPY_OBJ, pivot):
        bpy.context.scene.tool_settings.transform_pivot_point = pivot
