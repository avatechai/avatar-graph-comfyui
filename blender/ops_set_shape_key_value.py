import blender_node


class Mesh_SetShapeKeyValue(blender_node.ObjectOps):

    CUSTOM_NAME = "Set Shape Key Value"

    EXTRA_INPUT_TYPES = {
        "shape_key_name": ("STRING", {
            "multiline": False,
            "default": "my_shape_key",
        }),
        "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
    }

    def blender_process(self, bpy, BPY_OBJ, shape_key_name, value):
        # Check if the object has shape keys
        if BPY_OBJ.data.shape_keys:
            # Check if the specified shape key exists
            if shape_key_name in BPY_OBJ.data.shape_keys.key_blocks:
                BPY_OBJ.data.shape_keys.key_blocks[shape_key_name].value = float(value)
            else:
                print(f"The shape key {shape_key_name} does not exist on the object.")
        else:
            print("The object does not have any shape keys.")
        return (BPY_OBJ,)
