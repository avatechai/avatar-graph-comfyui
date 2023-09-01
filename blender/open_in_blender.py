import subprocess
import folder_paths
import os

blender_process_global = []

class OpenInBlender:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bpy_objs": ("BPY_OBJS",),
                "blender_path": ("STRING", {
                    "multiline": False,
                    "default": "blender"
                }),
                "shading": (["Material", "Solid", "Rendered", "Wireframe"],),
                "camera_location": ("VECTOR3D",),
                "camera_rotation": ("VECTOR3D",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process"

    OUTPUT_NODE = True

    CATEGORY = "mesh"

    def process(self, bpy_objs, blender_path, shading, camera_location, camera_rotation):
        import mathutils
        import global_bpy
        bpy = global_bpy.get_bpy()

        # Change shading mode and viewport
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = shading.upper()
                        rv3d = space.region_3d
                        rv3d.view_location = camera_location
                        rv3d.view_rotation = mathutils.Euler(camera_rotation).to_quaternion()

        # Open blender
        if hasattr(self, 'blender_process'):
            self.blender_process.kill()
            # remove from global list so it doesn't get garbage collected
            blender_process_global.remove(self.blender_process)

        # Save as .blend
        output_file = self.output_dir + '/tmp.blend'
        if os.path.exists(output_file):
            os.remove(output_file)
        bpy.ops.wm.save_as_mainfile(filepath=output_file)


        self.blender_process = subprocess.Popen([blender_path, output_file])
        # append to global list so it doesn't get garbage collected
        blender_process_global.append(self.blender_process)
        return { "ui" : {  } }

NODE_CLASS_MAPPINGS = {
    "OpenInBlender": OpenInBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenInBlender": "Open in Blender"
}

# detects when the python process is killed, and kills the blender process
import atexit
@atexit.register
def kill_blender_process():
    print('blender_process_global', blender_process_global)
    for process in blender_process_global:
        process.kill()