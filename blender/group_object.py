class GroupObject():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
                "bpy_objs_target_2": ("BPY_OBJS",),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, bpy_objs_target_2):
        import global_bpy
        bpy = global_bpy.get_bpy()

        target_object = bpy_objs_target[0]
        target_object2 = bpy_objs_target_2[0]

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "GroupObject": GroupObject
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroupObject": "Group Object"
}
