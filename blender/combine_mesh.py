class CombineMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "obj_1": ("BPY_OBJS",),
                "obj_2": ("BPY_OBJS",),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, obj_1, obj_2):
        import global_bpy
        bpy = global_bpy.get_bpy()

        override = bpy.context.copy()
        override["active_object"] = obj_1[0]
        override["selected_editable_objects"] = list([*obj_1, *obj_2])
        bpy.ops.object.join(override)
        joined_object = override["active_object"]
        return ([joined_object], )

NODE_CLASS_MAPPINGS = {
    "CombineMesh": CombineMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineMesh": "Combine Mesh"
}
