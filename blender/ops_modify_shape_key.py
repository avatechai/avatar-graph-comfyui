import blender_node


class Mesh_ModifyShapeKey(blender_node.ObjectOps):
    EXTRA_INPUT_TYPES = {
        "shape_key_name": ("STRING", {
            # True if you want the field to look like the one on the ClipTextEncode node
            "multiline": False,
            "default": "EyeBlinkLeft",  # default value
        }),
        "target_vertex_group": ("STRING", {
            "multiline": False,
            "default": ""
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
    }

    def blender_process(self, bpy, BPY_OBJ, shape_key_name, target_vertex_group, scale_x, scale_y, offset_x, offset_y, rotate, origin_offset_x, origin_offset_y, transform_radius, falloff):
        from mathutils import Vector, Matrix
        import math

        obj = BPY_OBJ

        # Create new shape key if its not in the object
        if shape_key_name not in BPY_OBJ.data.shape_keys.key_blocks:
            BPY_OBJ.shape_key_add(name=shape_key_name, from_mix=True)

        # get shape key
        sk = obj.data.shape_keys.key_blocks[shape_key_name]
        # print(obj, sk)
        # sk.interpolation = 'KEY_LINEAR'



        verts = obj.data.vertices

        # Create a new array of verts with sk as co and verts' groups
        new_verts = []
        for i, vert in enumerate(verts):
            print(i)
            print(sk.data[i].co)
            new_vert = {
                'co': sk.data[i].co,
                'groups': vert.groups
            }
            new_verts.append(new_vert)
        verts = new_verts

        # get vertex group

        # get the verts in the group

        if target_vertex_group == '':
            verts_in_group = verts
        else:
            vertex_group = obj.vertex_groups[target_vertex_group]
            verts_in_group = [v for v in verts if vertex_group.index in [
                vg.group for vg in v['groups']]]

        center = sum((vert['co'] for vert in verts_in_group),
                     Vector()) / len(verts_in_group)

        # apply origin offset
        center.x += origin_offset_x
        center.y += origin_offset_y

        # Calculate max distance from center to any vertex in the group
        max_distance = max(
            (vert['co'] - center).length for vert in verts_in_group)

        # Position each vert
        for i, vert in enumerate(verts):
            # return if vert is not in group
            if target_vertex_group != '' and vertex_group.index not in [vg.group for vg in vert['groups']]:
                continue
            # Get vector from center to vertex
            vec = vert['co'] - center

            # Calculate distance from origin and weight factor
            distance = vec.length
            # weight = max(0, 1 - distance / (max_distance * transform_radius))

            # if the vertex is outside the transform radius, skip it
            if (distance / max_distance) > transform_radius:
                continue

            # calculate weight
            weight = max(0, 1 - distance /
                         (transform_radius * max_distance)) * falloff
            weight = min(weight, 1)

            # print(scale_x, scale_y, weight)

            # Scale vector
            vec.x *= scale_x + (abs(scale_x - 1) * weight)
            vec.y *= scale_y + (abs(scale_y - 1) * weight)

            # # Offset vector
            vec.x += offset_x + (abs(scale_x - 1) * weight)
            vec.y += offset_y + (abs(scale_y - 1) * weight)

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
            print(i, vert['co'], new_pos, weight)

            sk.data[i].co = new_pos
