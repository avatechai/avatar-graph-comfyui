class GroupVertexInside():
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bpy": ("BPY",),
                "bpy_objs_target": ("BPY_OBJS",),
                "name": ("STRING", {
                    "multiline": False,
                    "default": "Group"
                }),
            },
        }

    RETURN_TYPES = ("BPY_OBJS",)
    RETURN_NAMES = ("bpy_objs",)

    FUNCTION = "process"

    CATEGORY = "mesh"

    def process(self, bpy_objs_target, name):
        import global_bpy
        bpy = global_bpy.get_bpy()

        target_object = bpy_objs_target[0]

        bpy.ops.object.mode_set(mode='OBJECT')

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # select only the target object
        bpy.context.view_layer.objects.active = target_object
        # enter enter edit mode and select all faces of the object to fill
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # select the vetex group's vertices
        bpy.ops.object.vertex_group_select()

        # get vertex group
        vertex_group = target_object.vertex_groups[name]

        verts = target_object.data.vertices

        # all_vertices = [(v.co.x, v.co.y, i) for i, v in target_object.data.vertices]

        # get the verts in the group
        verts_in_group = [v for v in verts if vertex_group.index in [vg.group for vg in v.groups]]

        # get all selected vertices's xy
        # get the vertices of the vertex group
        vertex_group_vertices = [(v.co.x, v.co.y) for v in verts_in_group]

        # get all vertices
        # all_vertices = [(v.co.x, v.co.y) for v in target_object.data.vertices]
        all_vertices_without_vertex_group = [(v.co.x, v.co.y, i) for i, v in enumerate(target_object.data.vertices) if v not in verts_in_group]

        from scipy.spatial import ConvexHull, Delaunay

        # create a convex hull of the vertex group's vertices
        hull = ConvexHull(vertex_group_vertices)

        # create a Delaunay triangulation of the convex hull
        tri = Delaunay(hull.points[hull.vertices])

        # get the indices of the vertices that are inside the convex hull
        inside_vertices_indices = [i for (x,y,i) in all_vertices_without_vertex_group if tri.find_simplex((x,y))>=0]
        
        print(inside_vertices_indices)
        print(len(vertex_group_vertices))

        bpy.ops.mesh.select_all(action='DESELECT')

        # select the vertices that are inside the convex hull
        # for i in inside_vertices_indices:
        #     target_object.data.vertices[i].select = True 

        # # select vertex from 1 to 10
        # for i in range(1, 100):
        #     target_object.data.vertices[i].select = True

        # create a new vertex group
        # bpy.context.object.vertex_groups.new(name="fuck")
        # bpy.ops.object.vertex_group_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
        # create a new vertex group
        # new_group = bpy.context.object.vertex_groups.new(name="new_group")
        vertex_group.add(inside_vertices_indices, weight=1.0, type='REPLACE')

        # assign the selected vertices to the new vertex group
        # new_group.add([v.index for v in target_object.data.vertices if v.select], weight=1.0, type='REPLACE')
        # new_group.add([20, 100], weight=1.0, type='REPLACE')
        # assign the selected vertices to the active vertex group
        bpy.ops.object.mode_set(mode='OBJECT')  


        return ([target_object],)


NODE_CLASS_MAPPINGS = {
    "GroupVertexInside": GroupVertexInside
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroupVertexInside": "Group Vertex Inside"
}
