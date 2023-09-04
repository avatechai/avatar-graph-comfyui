import folder_paths
import os

class ExportGLTF():
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bpy_objects": ("BPY_OBJS",),
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "out"
                }),
                "model_type": (["GLB", "GLTF_EMBEDDED"],),
                "write_mode": (["Overwrite", "Increment"],),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process"

    OUTPUT_NODE = True

    CATEGORY = "mesh"

    def process(self, bpy_objects, filename, model_type, write_mode):
        import global_bpy
        bpy = global_bpy.get_bpy()
        # print(bpy, bpy_objects)
        
        # deselect all objects
        override = bpy.context.copy()
        override["selected_objects"] = list(bpy_objects)
        override["active_object"] = list(bpy_objects)[0]

        for obj in bpy_objects:
            obj.select_set(True)

        filepath = self.output_dir + "/" + filename + '.' + ("glb" if model_type == "GLB" else "gltf")

        if write_mode == "Increment":
            count = 0
            # while file exists, increment count
            while os.path.exists(self.output_dir + "/" + filename + '_' + str(count) + '.' + ("glb" if model_type == "GLB" else "gltf")):
                count += 1

            filepath = self.output_dir + "/" + filename + '_' + str(count) + '.' + ("glb" if model_type == "GLB" else "gltf")
        
        with bpy.context.temp_override(**override):
            bpy.ops.export_scene.gltf(filepath=filepath, export_format=model_type, use_selection=True)
            print(filepath)

        return { "ui" : { "gltfFilename": { filepath.replace(f"{self.output_dir}/", "") } } }

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ExportGLTF": ExportGLTF
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExportGLTF": "ExportGLTF"
}
