import blender_node


class Mesh_JoinMesh(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "BPY_OBJ2": (blender_node.BPY_OBJ,)
    }
    
    CUSTOM_NAME = "Join Meshes"

    def blender_process(self, bpy, BPY_OBJ, **props):
        
        prop_values = props.values()
        
        override = bpy.context.copy()
        override["active_object"] = BPY_OBJ
        override["selected_editable_objects"] = list(prop_values) + [BPY_OBJ]
        bpy.ops.object.join(override)

        return (BPY_OBJ,)

