class CreateVertexGroup():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_target": ("BPY_OBJS",),
                "name": ("STRING", {
                    "multiline": False,
                    "default": "Group"
                }),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, name):
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

        # create vertex group and assign all selected faces to it (the name of the vertex group is the same as the name of the object)
        bpy.context.object.vertex_groups.new(name=name)
        bpy.ops.object.vertex_group_assign()

        # bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "CreateVertexGroup": CreateVertexGroup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateVertexGroup": "Vertex Group (Old)"
}
