import blender_node
from mesh_utils import genreate_mesh_from_texture

class Object_CreateMeshLayer(blender_node.ObjectOps):

    BASE_INPUT_TYPES = {}
    
    CUSTOM_NAME = "Create Mesh Layer"

    EXTRA_INPUT_TYPES = {
        "image": ("IMAGE",),
        "convex_hull": ("BOOLEAN", {"default": True}),
        # "face_threshold": ("FLOAT", {"display": "number", "default": 0.7}),
        "shape_threshold": ("FLOAT", {"display": "number", "default": 0.7}),
        "mesh_layer_name": ("STRING", {"default": "mesh_layer"}),
        "scale_x": ("FLOAT", {"display": "number", "default": 1}),
        "scale_y": ("FLOAT", {"display": "number", "default": 1}),
        "extrude_x": ("FLOAT", {"display": "number", "default": 0}),
        "extrude_y": ("FLOAT", {"display": "number", "default": 0}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
    }

    RETURN_TYPES = ("BPY_OBJ", "IMAGE")

    def blender_process(self, bpy, image, convex_hull, shape_threshold, mesh_layer_name, scale_x,scale_y , extrude_x, extrude_y, seed):
        image, BPY_OBJ = genreate_mesh_from_texture(bpy, image)

        bpy.context.view_layer.objects.active = BPY_OBJ

        self.edit_mode(bpy)

        if convex_hull:
            bpy.ops.mesh.convex_hull(
                delete_unused=True, use_existing_faces=True,
                shape_threshold=shape_threshold,
                # face_threshold=face_threshold
                face_threshold=0.7
            )

        bpy.ops.mesh.delete(type='EDGE_FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.edge_face_add()

        bpy.ops.transform.resize(value=(float(scale_x), float(scale_y), 1))

        bpy.context.object.vertex_groups.new(name=mesh_layer_name)
        bpy.ops.object.vertex_group_assign()

        if extrude_x != 0 or extrude_y != 0:
            bpy.ops.mesh.extrude_region_move()
            bpy.ops.object.vertex_group_remove_from()
            bpy.ops.transform.resize(value=(extrude_x, extrude_y, 0))
            bpy.ops.mesh.delete(type='ONLY_FACE')

        self.object_mode(bpy)
        return (BPY_OBJ, image)
