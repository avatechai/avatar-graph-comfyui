import folder_paths
import os
import requests
import base64
from mesh_utils import export_gltf

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
            "optional": {
                "blendshapes": ("blendshapes",),
            },
            "hidden": {"endpoint": "ENDPOINT", "token": "TOKEN", "baseModelId": "BASE_MODEL_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "process"

    OUTPUT_NODE = True

    CATEGORY = "mesh"

    def process(self, bpy_objects, filename, model_type, write_mode, blendshapes, endpoint=None, token=None, baseModelId=None):
        filepath = export_gltf(
            self.output_dir, bpy_objects, filename, model_type, write_mode
        )

        if endpoint != None and token != None and baseModelId != None:
            # Read binary file
            with open(filepath, "rb") as f:
                data = f.read()

            # Convert binary data to base64 string to ensure safe transmission
            data_string = base64.b64encode(data).decode()

            # Make the POST request
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                },
                json={
                    "modelStr": data_string,
                    "baseModelId": baseModelId,
                    "format": "threejs",
                },
            )
            if response.status_code != 200:
                return { "ui" : { "status_code": { response.status_code } } }
            model_id = response.json()["model_id"]
            return { "ui" : { "model_id": { model_id }, "status_code": { response.status_code } } }
        return { "ui" : { "gltfFilename": { filepath.replace(f"{self.output_dir}/", "") }, "blendshapes": { blendshapes } } }

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ExportGLTF": ExportGLTF
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExportGLTF": "ExportGLTF"
}
