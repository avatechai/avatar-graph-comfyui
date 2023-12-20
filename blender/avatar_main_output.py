import platform
import blender_node
from mesh_utils import assign_texture, open_in_blender as open_blender, export_gltf
import folder_paths

global_blender_path = ''

def get_os():
    os_name = platform.system()
    if os_name == "Darwin":
        return "mac"
    elif os_name == "Linux":
        return "linux"
    else:
        return "other"


os_type = get_os()

if os_type == "mac":
    global_blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
elif os_type == "linux":
    global_blender_path = "blender"


class AvatarMainOutput(blender_node.ObjectOps):
    def __init__(self):
        self.my_blender_process = None
        self.output_dir = folder_paths.get_output_directory()

    EXTRA_INPUT_TYPES = {
        "BPY_OBJS": ("BPY_OBJS",),
        "open_in_blender": ("BOOLEAN", {
            "default": False
        }),
        "auto_save": ("BOOLEAN", {
            "default": False
        }),
        "blender_path_override": ("STRING", {
            "multiline": False,
            "default": ''
        }),

        "filename": ("STRING", {
            "multiline": False,
            "default": "out"
        }),
        "model_type": (["AVA","GLB", "GLTF_EMBEDDED"],),
        "write_mode": (["Overwrite", "Increment"],),

        "SHAPE_FLOW": ("SHAPE_FLOW",),
    }

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def blender_process(self, bpy, BPY_OBJ=None, BPY_OBJS=None, open_in_blender=False, auto_save=False, blender_path_override='', filename='', model_type='', write_mode='', SHAPE_FLOW=''):
        if open_in_blender:
            p = blender_path_override if blender_path_override else global_blender_path
            output_file = self.output_dir + '/tmp.blend'
            a = open_blender(self.my_blender_process, blender_path=p,
                            output_file=output_file)
            self.my_blender_process = a
            

        objs = BPY_OBJS if BPY_OBJS else [BPY_OBJ]

        filepath = export_gltf(
            self.output_dir, objs, filename, model_type, write_mode, SHAPE_FLOW
        )

        import global_bpy
        global_bpy.set_should_reset_scene(True)

        return {
            "ui": {
                "gltfFilename": {filepath.replace(f"{self.output_dir}/", "")}, 
                "files": [{
                    "filename": filepath.replace(f"{self.output_dir}/", ""),
                    "type": "model/gltf+json",
                },],
                "SHAPE_FLOW": {SHAPE_FLOW}, 
                "auto_save": {'true' if auto_save else 'false'}, 
            }
        }
