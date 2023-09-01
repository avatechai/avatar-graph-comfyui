class KnifeProjection:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy_objs_knife": ("BPY_OBJS",),
                "bpy_objs_target": ("BPY_OBJS",),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_knife, bpy_objs_target):
        import mathutils
        import math
        import global_bpy
        bpy = global_bpy.get_bpy()

        knife_object = bpy_objs_knife[0]
        target_object = bpy_objs_target[0]

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = target_object
        # knife_object.select_set(True)
        # target_object.select_set(True)

        # bpy.ops.object.mode_set(mode='EDIT')

        area = next(
            area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        space = next(space for space in area.spaces if space.type == 'VIEW_3D')
        region = next(
            region for region in area.regions if region.type == 'WINDOW')

        # space.region_3d.view_perspective = 'ORTHO'
        # space.region_3d.view_rotation = mathutils.Euler(
        #     (0, 0, 0)).to_quaternion()

        # set blender to orthographic view without using numpad

        # bpy.ops.view3d.view_axis(type='TOP')

        override = bpy.context.copy()

        # override["active_object"] = target_object
        # override["edit_object"] = target_object
        # override["selected_objects"] = [knife_object, target_object]
        # override["selected_editable_objects"] = [knife_object, target_object]

        override["region"] = region
        override["area"] = area
        override["space"] = space

        override["active_object"] = target_object
        override["edit_object"] = target_object
        override["selected_objects"] = [knife_object, target_object]
        override["selected_editable_objects"] = [knife_object, target_object]

        knife_object.select_set(False)
        target_object.select_set(True)
        # move the kinfe object up a bit
        bpy.ops.transform.translate(value=(0, 0, -0.1))
        bpy.ops.object.mode_set(mode='EDIT')
        knife_object.select_set(False)

        # Get the viewport rotation
        viewport_rotation = space.region_3d.view_rotation

        with bpy.context.temp_override(**override):
            # # Create a rotation matrix representing a top-down view
            # rotation_matrix = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X')

            # # Create a rotation matrix to make the object face the user
            # rotation_matrix_face_user = mathutils.Matrix.Rotation(math.radians(180.0), 4, 'Z')

            # # Create a rotation matrix for the viewport rotation
            # rotation_matrix_viewport = viewport_rotation.to_matrix().to_4x4()

            # # Remember the original locations
            # original_location_knife = knife_object.location.copy()
            # original_location_target = target_object.location.copy()

            # # Move the objects to the origin
            # knife_object.location = mathutils.Vector((0, 0, 0))
            # target_object.location = mathutils.Vector((0, 0, 0))

            # # Rotate the objects
            # knife_object.matrix_world = rotation_matrix @ rotation_matrix_face_user @ rotation_matrix_viewport @ knife_object.matrix_world
            # target_object.matrix_world = rotation_matrix @ rotation_matrix_face_user @ rotation_matrix_viewport @ target_object.matrix_world

            bpy.ops.view3d.view_axis(type='TOP', align_active=False) 
            bpy.ops.view3d.view_persportho()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            bpy.ops.mesh.knife_project()

            # # Rotate the objects back
            # knife_object.matrix_world = rotation_matrix.inverted() @ knife_object.matrix_world
            # target_object.matrix_world = rotation_matrix.inverted() @ target_object.matrix_world

            # # Move the objects back to their original locations
            # knife_object.location = original_location_knife
            # target_object.location = original_location_target

        bpy.ops.object.mode_set(mode='OBJECT')

        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "KnifeProjection": KnifeProjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KnifeProjection": "Knife Projection"
}
