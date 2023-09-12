import blender_node


class Object_VertexGroupNewWithName(blender_node.EditOps):
    EXTRA_INPUT_TYPES = {
        "name": ("STRING", {
            "multiline": False,
            "default": "Group"
        }),
        "assign_selected": ("BOOLEAN", {
            "default": True
        }),
    }

    def blender_process(self, bpy, BPY_OBJ, name, assign_selected):
        bpy.context.object.vertex_groups.new(name=name)
        if assign_selected:
            bpy.ops.object.vertex_group_assign()