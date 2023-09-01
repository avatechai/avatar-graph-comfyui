class UV_Unwrap():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target):
        import global_bpy
        bpy = global_bpy.get_bpy()

        target_object = bpy_objs_target[0]

        bpy.context.view_layer.objects.active = target_object
        bpy.ops.object.mode_set(mode='OBJECT')
        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # select only the target object
        bpy.context.view_layer.objects.active = target_object
        # enter enter edit mode and select all faces of the object to fill
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # perform a cube projection unwrap
        bpy.ops.uv.cube_project(cube_size=1.0, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=True)

        # bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')


        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "UV_Unwrap": UV_Unwrap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UV_Unwrap": "UV Unwrap"
}
