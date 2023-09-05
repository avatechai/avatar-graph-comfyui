class TransfromObject():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
                "x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),

            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, x, y, z):
        import global_bpy
        bpy = global_bpy.get_bpy()

        target_object = bpy_objs_target[0]

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = target_object

        # move the object in level transformation
        target_object.location.x += x
        target_object.location.y += y
        target_object.location.z += z

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "TransfromObject": TransfromObject
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransfromObject": "Transfrom Object"
}
