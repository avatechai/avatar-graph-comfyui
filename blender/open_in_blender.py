import subprocess
import folder_paths
import os
from mesh_utils import open_in_blender


class OpenInBlender:
    def __init__(self):
        self.my_blender_process = None
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
        output_file = self.output_dir + '/tmp.blend'
        p = open_in_blender(self.my_blender_process, blender_path=blender_path, output_file=output_file, camera_location=camera_location,
                        camera_rotation=camera_rotation, shading=shading)
        self.my_blender_process = p
        return {"ui": {}}


NODE_CLASS_MAPPINGS = {
    "OpenInBlender": OpenInBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenInBlender": "Open in Blender"
}
