class ApplyMeshTransformAsShapeKey:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bpy_objs": ("BPY_OBJS",),
                "shape_key_name": ("STRING", {
                    # True if you want the field to look like the one on the ClipTextEncode node
                    "multiline": False,
                    "default": "EyeBlinkLeft",  # default value
                }),
                "target_vertex_group": ("STRING", {
                    "multiline": False,
                    "default": "Group"
                }),
                "scale_x": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                
                "rotate": ("FLOAT", {"default": 0, "min": -360, "max": 360.0, "step": 0.01, "display": "number"}),

                "origin_offset_x": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
                "origin_offset_y": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),

                "transform_radius": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01, "display": "number"}),
                "falloff": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "number"}),
            },
        }

    RETURN_TYPES = ("BPY_OBJS", )
    RETURN_NAMES = ("bpy_objs", )

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs, shape_key_name, target_vertex_group, scale_x, scale_y, offset_x, offset_y, rotate, origin_offset_x, origin_offset_y, transform_radius, falloff):
        from mathutils import Vector, Matrix
        import math
        import global_bpy
        bpy = global_bpy.get_bpy()

        obj = bpy_objs[0]

        # Create new shape key
        sk = obj.shape_key_add(name=shape_key_name)
        sk.interpolation = 'KEY_LINEAR'

        verts = obj.data.vertices

        # get vertex group
        vertex_group = obj.vertex_groups[target_vertex_group]

        # get the verts in the group
        verts_in_group = [v for v in verts if vertex_group.index in [
            vg.group for vg in v.groups]]

        # position each vert
        # for i in range(len(verts)):
        # Calculate center of mesh
        # center = sum((vert.co for vert in verts), Vector()) / len(verts)
        center = sum((vert.co for vert in verts_in_group),
                     Vector()) / len(verts_in_group)

        # apply origin offset
        center.x += origin_offset_x
        center.y += origin_offset_y
        # apply scale relative to object center

        # Calculate max distance from center to any vertex in the group
        max_distance = max(
            (vert.co - center).length for vert in verts_in_group)

        # Position each vert
        for i, vert in enumerate(verts):
            # return if vert is not in group
            if vertex_group.index not in [vg.group for vg in vert.groups]:
                continue
            # Get vector from center to vertex
            vec = vert.co - center

            # Calculate distance from origin and weight factor
            distance = vec.length
            # weight = max(0, 1 - distance / (max_distance * transform_radius))

            # if the vertex is outside the transform radius, skip it
            if (distance / max_distance) > transform_radius:
                continue

            # calculate weight
            weight = max(0, 1 - distance / (transform_radius * max_distance)) * falloff
            weight = min(weight, 1)

            # print(scale_x, scale_y, weight)

            # Scale vector
            vec.x *= scale_x + (abs(scale_x - 1) * weight)
            vec.y *= scale_y + (abs(scale_y - 1) * weight)

            # # Offset vector
            vec.x += offset_x * weight
            vec.y += offset_y * weight

            # Rotate vector
            # rotate_right_rad = math.radians(rotate_right)
            rotate_rad = math.radians(rotate)

            # Create rotation matrices for the right and left rotations
            # rotation_matrix_right = Matrix.Rotation(rotate_right_rad, 4, 'X')
            rotation_matrix = Matrix.Rotation(rotate_rad, 4, 'Z')

            # Apply the rotation matrices to the vector
            # vec.rotate(rotation_matrix_right)
            vec.rotate(rotation_matrix)

            # Calculate new position
            new_pos = center + vec

            # Apply new position
            sk.data[i].co = new_pos

        return (bpy_objs, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ApplyMeshTransformAsShapeKey": ApplyMeshTransformAsShapeKey
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyMeshTransformAsShapeKey": "Transform Shape Key"
}
