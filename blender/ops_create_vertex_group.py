import blender_node


class CreateVertexGroup(blender_node.EditOps):
    EXTRA_INPUT_TYPES = {
        "name": ("STRING", {
            "multiline": False,
            "default": "Group"
        }),
    }

    def blender_process(self, bpy, BPY_OBJ, name, **props):
        bpy.context.object.vertex_groups.new(name=name)