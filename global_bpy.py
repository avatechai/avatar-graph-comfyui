import bpy
global_bpy = bpy

def get_bpy(): 
    global global_bpy
    return global_bpy

def reset_bpy():
    global global_bpy
    bpy = global_bpy
    # not using this cause this cause seg fault frequently
    # global_bpy.ops.wm.read_factory_settings(use_empty=True)

    # enter object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Delete all objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Delete all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Delete all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Delete all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)