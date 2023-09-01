class ScaleVertex():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
                "extrude": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, extrude):
        import global_bpy
        bpy = global_bpy.get_bpy()
        
        if len(bpy_objs_target) == 0:
            # throw error
            return

        target_object = bpy_objs_target[0]

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # select only the target object
        bpy.context.view_layer.objects.active = target_object
        # enter enter edit mode and select all faces of the object to fill
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        bpy.ops.transform.resize(
            value=(extrude, extrude, 1), constraint_axis=(False, False, False))

        # remove the extruded vertex from the vertex group "name"
        # bpy.ops.object.vertex_group_remove_from()

        # bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)

    
NODE_CLASS_MAPPINGS = {
    "ScaleVertex": ScaleVertex
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleVertex": "Scale Vertex"
}
