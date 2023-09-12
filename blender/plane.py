class Plane:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "division": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "display": "number" 
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # For disabling cache
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, division, seed):
        import global_bpy
        bpy = global_bpy.get_bpy()

        # Define coordinates for the plane to object mode 
        if bpy.context.active_object is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        coords = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)]

        # Create a mesh object
        mesh = bpy.data.meshes.new(name="PlaneMesh")
        mesh.from_pydata(coords,[],[(0,1,3,2)])

        # Create a new object with the mesh
        object_a = bpy.data.objects.new(name="PlaneObject", object_data=mesh)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.collection.objects.link(object_a)
        bpy.context.view_layer.objects.active = object_a
        object_a.select_set(True)

        # Invert the Y axis of the UV to fix the vertical inversion
        bpy.ops.transform.resize(value=(-1, -1, 1))
        
        # Create a new UV map
        uv_map = mesh.uv_layers.new(name="newUV")
        mesh.uv_layers.active = uv_map

        # Unwrap the mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Subdivide the plane
        if division > 0:
            original_mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.subdivide(number_cuts=division)
            bpy.ops.object.mode_set(mode=original_mode)

        sk_basis = object_a.shape_key_add(name='Basis')

        return ([object_a],)

NODE_CLASS_MAPPINGS = {
    "Plane": Plane
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Plane": "Plane (Old)"
}
