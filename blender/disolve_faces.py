class DissolveFaces():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
                 "target_vertex_group": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, target_vertex_group):
        import global_bpy
        bpy = global_bpy.get_bpy()

        target_object = bpy_objs_target[0]

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # select only the target object
        bpy.context.view_layer.objects.active = target_object
        # enter enter edit mode and select all faces of the object to fill
        bpy.ops.object.mode_set(mode='EDIT')

        bpy.ops.mesh.select_all(action='DESELECT')

        # if we have vertex group, select that instead
        if target_vertex_group and target_vertex_group.strip() != "":
            group = target_object.vertex_groups.get(target_vertex_group)
            if group:
                bpy.ops.object.vertex_group_set_active(group=group.name)
                bpy.ops.object.vertex_group_select()
        else:
            bpy.ops.mesh.select_all(action='SELECT')

        # dissolve the faces
        # bpy.ops.mesh.dissolve_faces()
        bpy.ops.mesh.delete(type='ONLY_FACE')
        # bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "DissolveFaces": DissolveFaces
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DissolveFaces": "Dissolve Faces (Old)"
}
