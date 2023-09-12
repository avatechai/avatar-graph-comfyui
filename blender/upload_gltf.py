import requests
import base64


class UploadGLTF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": (
                    "STRING",
                    {"multiline": False, "default": "https://labs.avatech.ai/api/glb"},
                ),
                "filepath": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {"token": "TOKEN", "baseModelId": "BASE_MODEL_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "upload"

    CATEGORY = "blender"

    OUTPUT_NODE = True

    def upload(self, endpoint, filepath, token=None, baseModelId=None):
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
        model_id = response.json()["model_id"]
        return {"ui": {"model_id": [model_id]}}


NODE_CLASS_MAPPINGS = {"UploadGLTF": UploadGLTF}

NODE_DISPLAY_NAME_MAPPINGS = {"UploadGLTF": "Upload GLTF"}
