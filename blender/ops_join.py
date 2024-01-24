import blender_node


class Mesh_JoinMesh(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {"BPY_OBJ2": (blender_node.BPY_OBJ,)}

    CUSTOM_NAME = "Join Meshes"

    def blender_process(self, bpy, BPY_OBJ, **props):
        prop_values = props.values()
        for obj in list(prop_values) + [BPY_OBJ]:
            if obj is not None:
                obj.select_set(True)
        if bpy.context.view_layer.objects is not None:
            bpy.context.view_layer.objects.active = BPY_OBJ
        bpy.ops.object.join()

        return (BPY_OBJ,)
