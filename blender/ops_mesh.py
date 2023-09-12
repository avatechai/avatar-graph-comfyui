import blender_node

mesh_api = ['attribute_set', 'average_normals', 'beautify_fill', 'bevel', 'bisect', 'blend_from_shape', 'bridge_edge_loops', 'colors_reverse', 'colors_rotate', 'convex_hull', 'customdata_bevel_weight_edge_add', 'customdata_bevel_weight_edge_clear', 'customdata_bevel_weight_vertex_add', 'customdata_bevel_weight_vertex_clear', 'customdata_crease_edge_add', 'customdata_crease_edge_clear', 'customdata_crease_vertex_add', 'customdata_crease_vertex_clear', 'customdata_custom_splitnormals_add', 'customdata_custom_splitnormals_clear', 'customdata_mask_clear', 'customdata_skin_add', 'customdata_skin_clear', 'decimate', 'delete', 'delete_edgeloop', 'delete_loose', 'dissolve_degenerate', 'dissolve_edges', 'dissolve_faces', 'dissolve_limited', 'dissolve_mode', 'dissolve_verts', 'dupli_extrude_cursor', 'duplicate', 'duplicate_move', 'edge_collapse', 'edge_face_add', 'edge_rotate', 'edge_split', 'edgering_select', 'edges_select_sharp', 'extrude_context', 'extrude_context_move', 'extrude_edges_indiv', 'extrude_edges_move', 'extrude_faces_indiv', 'extrude_faces_move', 'extrude_manifold', 'extrude_region', 'extrude_region_move', 'extrude_region_shrink_fatten', 'extrude_repeat', 'extrude_vertices_move', 'extrude_verts_indiv', 'face_make_planar', 'face_set_extract', 'face_split_by_edges', 'faces_mirror_uv', 'faces_select_linked_flat', 'faces_shade_flat', 'faces_shade_smooth', 'fill', 'fill_grid', 'fill_holes', 'flip_normals', 'flip_quad_tessellation', 'hide', 'inset', 'intersect', 'intersect_boolean', 'knife_project', 'knife_tool', 'loop_multi_select', 'loop_select', 'loop_to_region', 'loopcut', 'loopcut_slide', 'mark_freestyle_edge', 'mark_freestyle_face', 'mark_seam', 'mark_sharp', 'merge', 'merge_normals',
            'mod_weighted_strength', 'normals_make_consistent', 'normals_tools', 'offset_edge_loops', 'offset_edge_loops_slide', 'paint_mask_extract', 'paint_mask_slice', 'point_normals', 'poke', 'polybuild_delete_at_cursor', 'polybuild_dissolve_at_cursor', 'polybuild_extrude_at_cursor_move', 'polybuild_face_at_cursor', 'polybuild_face_at_cursor_move', 'polybuild_split_at_cursor', 'polybuild_split_at_cursor_move', 'polybuild_transform_at_cursor', 'polybuild_transform_at_cursor_move',  'quads_convert_to_tris', 'region_to_loop', 'remove_doubles', 'reveal', 'rip', 'rip_edge', 'rip_edge_move', 'rip_move', 'screw', 'select_all', 'select_axis', 'select_face_by_sides', 'select_interior_faces', 'select_less', 'select_linked', 'select_linked_pick', 'select_loose', 'select_mirror', 'select_mode', 'select_more', 'select_next_item', 'select_non_manifold', 'select_nth', 'select_prev_item', 'select_random', 'select_similar', 'select_similar_region', 'select_ungrouped', 'separate', 'set_normals_from_faces', 'shape_propagate_to_all', 'shortest_path_pick', 'shortest_path_select', 'smooth_normals', 'solidify', 'sort_elements', 'spin', 'split', 'split_normals', 'subdivide', 'subdivide_edgering', 'symmetrize', 'symmetry_snap', 'tris_convert_to_quads', 'unsubdivide', 'uv_texture_add', 'uv_texture_remove', 'uvs_reverse', 'uvs_rotate', 'vert_connect', 'vert_connect_concave', 'vert_connect_nonplanar', 'vert_connect_path', 'vertices_smooth', 'vertices_smooth_laplacian', 'wireframe']

object_api = ['add', 'add_named', 'align', 'anim_transforms_to_deltas', 'armature_add', 'assign_property_defaults', 'bake', 'bake_image', 'camera_add', 'clear_override_library', 'collection_add', 'collection_external_asset_drop', 'collection_instance_add', 'collection_link', 'collection_objects_select', 'collection_remove', 'collection_unlink', 'constraint_add', 'constraint_add_with_targets', 'constraints_clear', 'constraints_copy', 'convert', 'correctivesmooth_bind', 'curves_empty_hair_add', 'curves_random_add', 'data_instance_add', 'data_transfer', 'datalayout_transfer', 'delete', 'drop_geometry_nodes', 'drop_named_image', 'drop_named_material', 'duplicate', 'duplicate_move', 'duplicate_move_linked', 'duplicates_make_real', 'editmode_toggle', 'effector_add', 'empty_add', 'explode_refresh', 'face_map_add', 'face_map_assign', 'face_map_deselect', 'face_map_move', 'face_map_remove', 'face_map_remove_from', 'face_map_select', 'forcefield_toggle', 'geometry_node_tree_copy_assign', 'geometry_nodes_input_attribute_toggle', 'geometry_nodes_move_to_nodes', 'gpencil_add', 'gpencil_modifier_add', 'gpencil_modifier_apply', 'gpencil_modifier_copy', 'gpencil_modifier_copy_to_selected', 'gpencil_modifier_move_down', 'gpencil_modifier_move_to_index', 'gpencil_modifier_move_up', 'gpencil_modifier_remove', 'hide_collection', 'hide_render_clear_all', 'hide_view_clear', 'hide_view_set', 'hook_add_newob', 'hook_add_selob', 'hook_assign', 'hook_recenter', 'hook_remove', 'hook_reset', 'hook_select', 'instance_offset_from_cursor', 'instance_offset_from_object', 'instance_offset_to_cursor', 'isolate_type_render', 'join', 'join_shapes', 'join_uvs', 'laplaciandeform_bind', 'light_add', 'lightprobe_add', 'lightprobe_cache_bake', 'lightprobe_cache_free', 'lineart_bake_strokes', 'lineart_bake_strokes_all', 'lineart_clear', 'lineart_clear_all', 'link_to_collection', 'load_background_image', 'load_reference_image', 'location_clear', 'make_dupli_face', 'make_links_data', 'make_links_scene', 'make_local', 'make_override_library', 'make_single_user', 'material_slot_add', 'material_slot_assign', 'material_slot_copy', 'material_slot_deselect', 'material_slot_move', 'material_slot_remove', 'material_slot_remove_unused', 'material_slot_select', 'meshdeform_bind', 'metaball_add', 'mode_set', 'mode_set_with_submode', 'modifier_add', 'modifier_apply', 'modifier_apply_as_shapekey', 'modifier_convert', 'modifier_copy', 'modifier_copy_to_selected', 'modifier_move_down',
              'modifier_move_to_index', 'modifier_move_up', 'modifier_remove', 'modifier_set_active', 'move_to_collection', 'multires_base_apply', 'multires_external_pack', 'multires_external_save', 'multires_higher_levels_delete', 'multires_rebuild_subdiv', 'multires_reshape', 'multires_subdivide', 'multires_unsubdivide', 'ocean_bake', 'origin_clear', 'origin_set', 'parent_clear', 'parent_inverse_apply', 'parent_no_inverse_set', 'parent_set', 'particle_system_add', 'particle_system_remove', 'paths_calculate', 'paths_clear', 'paths_update', 'paths_update_visible', 'pointcloud_add', 'posemode_toggle', 'quadriflow_remesh', 'quick_explode', 'quick_fur', 'quick_liquid', 'quick_smoke', 'randomize_transform', 'reset_override_library', 'rotation_clear', 'scale_clear', 'select_all', 'select_by_type', 'select_camera', 'select_grouped', 'select_hierarchy', 'select_less', 'select_linked', 'select_mirror', 'select_more', 'select_pattern', 'select_random', 'select_same_collection', 'shade_flat', 'shade_smooth', 'shaderfx_add', 'shaderfx_copy', 'shaderfx_move_down', 'shaderfx_move_to_index', 'shaderfx_move_up', 'shaderfx_remove', 'shape_key_add', 'shape_key_clear', 'shape_key_mirror', 'shape_key_move', 'shape_key_remove', 'shape_key_retime', 'shape_key_transfer', 'simulation_nodes_cache_bake', 'simulation_nodes_cache_calculate_to_frame', 'simulation_nodes_cache_delete', 'skin_armature_create', 'skin_loose_mark_clear', 'skin_radii_equalize', 'skin_root_mark', 'speaker_add', 'subdivision_set', 'surfacedeform_bind', 'text_add', 'track_clear', 'track_set', 'transfer_mode', 'transform_apply', 'transform_axis_target', 'transform_to_mouse', 'transforms_to_deltas', 'unlink_data', 'vertex_group_add', 'vertex_group_assign', 'vertex_group_assign_new', 'vertex_group_clean', 'vertex_group_copy', 'vertex_group_copy_to_selected', 'vertex_group_deselect', 'vertex_group_invert', 'vertex_group_levels', 'vertex_group_limit_total', 'vertex_group_lock', 'vertex_group_mirror', 'vertex_group_move', 'vertex_group_normalize', 'vertex_group_normalize_all', 'vertex_group_quantize', 'vertex_group_remove', 'vertex_group_remove_from', 'vertex_group_select', 'vertex_group_set_active', 'vertex_group_smooth', 'vertex_group_sort', 'vertex_parent_set', 'vertex_weight_copy', 'vertex_weight_delete', 'vertex_weight_normalize_active_vertex', 'vertex_weight_paste', 'vertex_weight_set_active', 'visual_transform_apply', 'volume_add', 'volume_import', 'voxel_remesh', 'voxel_size_edit']

create_primitive_shape_api = ['primitive_circle_add', 'primitive_cone_add', 'primitive_cube_add', 'primitive_cube_add_gizmo',
                              'primitive_cylinder_add', 'primitive_grid_add', 'primitive_ico_sphere_add', 'primitive_monkey_add', 'primitive_plane_add', 'primitive_torus_add', 'primitive_uv_sphere_add',]

transfrom_api = ['bbone_resize', 'bend', 'create_orientation', 'delete_orientation', 'edge_bevelweight', 'edge_crease', 'edge_slide', 'from_gizmo', 'mirror', 'push_pull', 'resize', 'rotate', 'rotate_normal',
                 'select_orientation', 'seq_slide', 'shear', 'shrink_fatten', 'skin_resize', 'tilt', 'tosphere', 'trackball', 'transform', 'translate', 'vert_crease', 'vert_slide', 'vertex_random', 'vertex_warp']

bpy_object_member = [('active_material_index', 0), ('active_shape_key_index', 2), ('add_rest_position_attribute', False), ('display_bounds_type', 'BOX'), ('display_type', 'TEXTURED'), ('empty_display_size', 1.0), ('empty_display_type', 'ARROWS'), ('empty_image_depth', 'DEFAULT'), ('empty_image_side', 'DOUBLE_SIDED'), ('hide_render', False), ('hide_select', False), ('hide_viewport', False), ('instance_faces_scale', 1.0), ('instance_type', 'NONE'), ('is_embedded_data', False), ('is_evaluated', False), ('is_from_instancer', False), ('is_from_set', False), ('is_holdout', False), ('is_instancer', False), ('is_library_indirect', False), ('is_missing', False), ('is_runtime_data', False), ('is_shadow_catcher', False), ('lightgroup', ''), ('lock_rotation_w', False), ('lock_rotations_4d', False), ('mode', 'OBJECT'), ('name', 'Cube'), ('name_full', 'Cube'), ('parent_bone', ''), ('parent_type', 'OBJECT'), ('pass_index', 0), ('rotation_mode', 'XYZ'), ('show_all_edges', False), ('show_axis', False), ('show_bounds', False), ('show_empty_image_only_axis_aligned', False), ('show_empty_image_orthographic', True), ('show_empty_image_perspective', True), ('show_in_front', False), ('show_instancer_for_render', True), ('show_instancer_for_viewport', True), ('show_name', False), ('show_only_shape_key', False), ('show_texture_space', False), ('show_transparent', False), ('show_wire', False), ('tag', False), ('track_axis', 'POS_Y'), ('type', 'MESH'), ('up_axis', 'Z'), ('use_camera_lock_parent', False), ('use_dynamic_topology_sculpting', False), ('use_empty_image_alpha', False), ('use_extra_user', False), ('use_fake_user', False), ('use_grease_pencil_lights', False), ('use_instance_faces_scale', False), ('use_instance_vertices_rotation', False), ('use_mesh_mirror_x', False), ('use_mesh_mirror_y', False), ('use_mesh_mirror_z', False), ('use_shape_key_edit_mode', False), ('use_simulation_cache', True), ('users', 1), ('visible_camera', True), ('visible_diffuse', True), ('visible_glossy', True), ('visible_shadow', True), ('visible_transmission', True), ('visible_volume_scatter', True)]

BLENDER_NODES = [
    blender_node.create_ops_class(
        blender_node.EditOps, 'ops.mesh.' + op, None, 'Mesh_') for op in mesh_api
] + [
    blender_node.create_ops_class(
        blender_node.ObjectOps, 'ops.object.' + op, None, 'Object_') for op in object_api
] + [
    blender_node.create_primitive_shape_class(
        blender_node.ObjectOps, 'ops.mesh.' + op, None, 'Mesh_') for op in create_primitive_shape_api
] + [
    blender_node.create_ops_class(
        blender_node.EditOps, 'ops.transform.' + op, None, 'Transform_') for op in transfrom_api
] + [
    blender_node.create_obj_setter_class(blender_node.ObjectOps, op) for op in bpy_object_member
]

# print(blender_node.print_blender_functions('ops','object'))
# print(blender_node.print_blender_functions('types.Object'))
# print(blender_node.print_blender_functions('context.object.vertex_groups.new'))
# print(blender_node.print_blender_functions('ops.transform'))


# Get the class itself, not an instance
# cls = bpy.context.object

# # Get all callable methods of the class
# methods = [(method_name, method,)
#            for method_name, method in inspect.getmembers(cls, )]
# # only keep those with that are string, int or float, boolean
# simple_setter = [(method_name, method,) for method_name, method in methods if not callable(
#     method) and not method_name.startswith("__") and type(method) in [str, int, float, bool]]

# simple_function = [(method_name, method,) for method_name, method in methods if callable(
#     method) and not method_name.startswith("__")]


# simple_function_with_args = []
# for method_name, method in simple_function:
#     print(dir(method), method.call())
#     try:
#         params = inspect.signature(method).parameters
#         simple_function_with_args.append((method_name, method, params))
#     except ValueError:
#         print(f"Cannot retrieve signature for {method_name}", type(method),)

# print(simple_function_with_args)

# print("Content of cls.shape_key_add:")
# for item in dir(cls.shape_key_add):
#     print(item, getattr(cls.shape_key_add, item))


# cls.shape_key_add

