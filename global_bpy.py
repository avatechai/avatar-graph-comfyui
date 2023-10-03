import bpy

global_bpy = bpy
should_reset_scene = True


def set_should_reset_scene(value):
    global should_reset_scene
    should_reset_scene = value


def get_bpy():
    global global_bpy, should_reset_scene
    if should_reset_scene:
        reset_bpy()
        should_reset_scene = False
    return global_bpy


def reset_bpy():
    global global_bpy
    bpy = global_bpy
    # not using this cause this cause seg fault frequently
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # # check if there's no object, return
    # if len(bpy.data.objects) == 0:
    #     return

    # # enter object mode
    # if bpy.context.mode != 'OBJECT':
    #     bpy.ops.object.mode_set(mode='OBJECT')

    # # Delete all objects
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete()

    # # Delete all meshes
    # for mesh in bpy.data.meshes:
    #     bpy.data.meshes.remove(mesh)

    # # Delete all materials
    # for material in bpy.data.materials:
    #     bpy.data.materials.remove(material)

    # # Delete all textures
    # for texture in bpy.data.textures:
    #     bpy.data.textures.remove(texture)
