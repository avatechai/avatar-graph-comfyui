class FillFace:
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

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # select only the target object
        bpy.context.view_layer.objects.active = target_object
        # enter enter edit mode and select all faces of the object to fill
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # fill the faces
        bpy.ops.mesh.fill()

        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "FillFace": FillFace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillFace": "Fill Face (Old)"
}
