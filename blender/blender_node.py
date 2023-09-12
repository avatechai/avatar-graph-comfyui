import inspect
import re
BPY_OBJS = "BPY_OBJS"
BPY_OBJ = "BPY_OBJ"

BPY_OBJS_TYPE = {
    BPY_OBJS: (BPY_OBJS,),
}


class ObjectOps:
    @classmethod
    def get_extra_input_types(cls, bpy):
        return cls.EXTRA_INPUT_TYPES

    @classmethod
    def get_base_input_types(cls, bpy):
        return cls.BASE_INPUT_TYPES

    EXTRA_INPUT_TYPES = {}
    BASE_INPUT_TYPES = {
        "BPY_OBJ": ("BPY_OBJ",)
    }

    @classmethod
    def INPUT_TYPES(cls):
        import global_bpy
        bpy = global_bpy.get_bpy()
        result = {
            "required": {},
            "optional": {
                **cls.get_base_input_types(bpy),
                **cls.get_extra_input_types(bpy)
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
            cls.__name__: re.sub("([a-z])([A-Z])", "\g<1> \g<2>", cls.__name__).replace('_', ' ')
        }

    RETURN_TYPES = ("BPY_OBJ",)
    FUNCTION = "process"
    CATEGORY = "blender"

    def process(self, **props):
        import global_bpy
        bpy = global_bpy.get_bpy()

        results = self.blender_process(bpy, **props)

        if results is None:
            print(results)
            if props.get("BPY_OBJ") != None:
                return (props["BPY_OBJ"], )
            else:
                return (bpy.context.view_layer.objects.active, )

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


def map_args(bpy, func):
    # print(func, type(func))
    rna_type = func.get_rna_type()
    args_dict = {}
    for prop in rna_type.properties:
        if prop.is_readonly:
            continue
        prop_type = str(prop.type)
        # print(prop.identifier, prop_type, prop, )
        prop_dict = {}
        if prop_type in ["INT", "FLOAT"]:
            if not prop.is_array:
                prop_dict.update(
                    {"min": prop.hard_min, "max": prop.hard_max, "default": prop.default})
            else:
                prop_type = "B_VECTOR" + str(prop.array_length)
                # prop_dict.update(
                #     {"default": prop.default_array})
        elif prop_type == "BOOLEAN":
            prop_dict.update({"default": prop.default})
        elif prop_type == "STRING":
            prop_dict.update(
                {"default": prop.default, "multiline": False})
        elif prop_type == "ENUM":
            enum_items = [item.identifier for item in prop.enum_items]
            args_dict[prop.identifier] = (enum_items,)

            # print(enum_items)

        if args_dict.get(prop.identifier) is None:
            args_dict[prop.identifier] = (prop_type, prop_dict)

    # print(args_dict)
    return args_dict


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    # capitalize the first letter of each component and join them together
    camel_str = ''.join(x.title() for x in components)
    return camel_str


def get_nested_attr(obj, attr_str):
    attrs = attr_str.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def print_blender_functions(path):
    import sys
    import os

    ag_path = os.path.join(os.path.dirname(__file__), '../')
    if ag_path not in sys.path:
        sys.path.append(ag_path)

    import global_bpy
    bpy = global_bpy.get_bpy()

    # add a an empty object

    print(dir(get_nested_attr(bpy, path)), type(get_nested_attr(bpy, path)))


def create_ops_class(cls, path, name=None, name_prefix=''):
    node_name = name_prefix + snake_to_camel(
        name if name is not None else path.split('.')[-1])
    # print(node_name)
    return type(
        node_name, (cls, object),
        {
            'get_extra_input_types': classmethod(
                lambda cls, bpy: {
                    **map_args(bpy, get_nested_attr(bpy, path))}
            ),
            'blender_process': lambda self, bpy, BPY_OBJ, **props:
                (None, get_nested_attr(bpy, path)(**props))[0]
        }
    )


def create_primitive_shape_class(cls, path, name=None, name_prefix=''):
    node_name = name_prefix + snake_to_camel(
        name if name is not None else path.split('.')[-1])
    # print(node_name)
    return type(
        node_name, (cls, object),
        {
            'BASE_INPUT_TYPES': {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            'get_extra_input_types': classmethod(
                lambda cls, bpy: {
                    **map_args(bpy, get_nested_attr(bpy, path))}
            ),
            'blender_process': lambda self, bpy, **props:
                (print(props), get_nested_attr(bpy, path)(), None)[2]
        }
    )


def assign_and_return(BPY_OBJ, name, value):
    BPY_OBJ[name] = value
    return None


def create_obj_setter_class(cls, item):
    name = item[0]
    item_type = {
        int: "INT",
        float: "FLOAT",
        str: "STRING",
        bool: "BOOLEAN",
    }[type(item[1])]

    node_name = 'ObjectSet_' + snake_to_camel(
        name if name is not None else path.split('.')[-1])

    ainput = {
        'value': (item_type, {"default": item[1]})
    }

    # print(item, item_type, type(item[1]), ainput)

    return type(
        node_name, (cls, object),
        {
            'get_extra_input_types': classmethod(
                lambda cls, bpy: ainput),
            'blender_process': lambda self, bpy, BPY_OBJ, **props:
                assign_and_return(BPY_OBJ, name, props['value'])
        }
    )
