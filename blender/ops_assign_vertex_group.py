import blender_node


class AssignVertexGroupOps(blender_node.EditOps):
    EXTRA_INPUT_TYPES = {
        "name": ("STRING", {
            "multiline": False,
            "default": "Group"
        }),
    }

    def blender_process(self, bpy, BPY_OBJ, name, **props):
        # activate the vertex group by name
        bpy.context.object.vertex_groups.active = bpy.context.object.vertex_groups[name]
        bpy.ops.object.vertex_group_assign()
