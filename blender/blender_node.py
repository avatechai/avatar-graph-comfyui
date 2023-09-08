BPY_OBJS = "BPY_OBJS"
BPY_OBJ = "BPY_OBJ"

BPY_OBJS_TYPE = {
    BPY_OBJS: (BPY_OBJS,),
}

class ObjectOps:
    EXTRA_INPUT_TYPES = {}
    BASE_INPUT_TYPES = {
        "BPY_OBJ": ("BPY_OBJ",)
    }

    @classmethod
    def INPUT_TYPES(cls):
        # print(cls.BASE_INPUT_TYPES)
        # print(cls.EXTRA_INPUT_TYPES)
        result = {
            "required": {
                **cls.BASE_INPUT_TYPES,
                **cls.EXTRA_INPUT_TYPES
            }
        }
        return result

    @classmethod
    def NODE_CLASS_MAPPINGS(cls):
        return {
            cls.__name__: cls
        }

    @classmethod
    def NODE_DISPLAY_NAME_MAPPINGS(cls):
        import re
        return {
            cls.__name__: re.sub("([a-z])([A-Z])","\g<1> \g<2>",cls.__name__)
        }

    RETURN_TYPES = ("BPY_OBJ",)
    FUNCTION = "process"
    CATEGORY = "blender"

    def process(self, **props):
        import global_bpy
        bpy = global_bpy.get_bpy()

        results = self.blender_process(bpy, **props)

        if results is None:
            return (props["BPY_OBJ"], )

        return results

    def blender_process(self, bpy, **props):
        pass


class EditOps(ObjectOps):
    def process(self, **props):
        import global_bpy
        bpy = global_bpy.get_bpy()

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = props[BPY_OBJ]

        bpy.ops.object.mode_set(mode='EDIT')
        results = self.blender_process(bpy, **props)
        if bpy.context.object.mode == 'EDIT' and len(bpy.data.objects) > 0:
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')

        if results is None:
            return (props["BPY_OBJ"], )

        return results
