class CleanFace():
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
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.mesh.delete(type='EDGE_FACE')
        bpy.ops.mesh.remove_doubles()
        # bpy.ops.mesh.delete(type='FACE')
        # bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.mesh.edge_face_add()
        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "CleanFace": CleanFace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CleanFace": "Clean Face"
}
