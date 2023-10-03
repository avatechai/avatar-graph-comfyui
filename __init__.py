"""
@author: Avatech Limited
@title: Avatar Graph
@nickname: Avatar Graph
@description: Include nodes for sam + bpy operation, that allows workflow creations for generative 2d character rig.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

import routes
import inspect
import sys
import importlib
import subprocess
import requests
import folder_paths
from folder_paths import add_model_folder_path, get_filename_list, get_folder_paths
from tqdm import tqdm

ag_path = os.path.join(os.path.dirname(__file__))

# pip_install = [sys.executable, '-m', 'pip', 'install']

# def ensure_pip_packages():
#     try:
#         # import bpy
#         import segment_anything
#     except Exception:
#         my_path = os.path.dirname(__file__)
#         requirements_path = os.path.join(my_path, "requirements.txt")
#         subprocess.check_call(pip_install + ['-r', requirements_path])

# ensure_pip_packages()


def get_python_files(path):
    return [f[:-3] for f in os.listdir(path) if f.endswith(".py")]


def append_to_sys_path(path):
    if path not in sys.path:
        sys.path.append(path)

folder_paths.folder_names_and_paths["sams"] = ([os.path.join(folder_paths.models_dir, "sams")], folder_paths.supported_pt_extensions)

def download_sam_model():
    model_dir = get_folder_paths("sams")[0]
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    add_model_folder_path("sams", model_dir)

    files = get_filename_list("sams")
    if len(files) == 0:
        print("Downloading sam model...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)

        with open(f"{model_dir}/sam_vit_h_4b8939.pth", "wb") as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit="KB",
                desc=url.split("/")[-1],
            ):
                f.write(chunk)


download_sam_model()

paths = ["blender", "sam"]
files = []

for path in paths:
    full_path = os.path.join(ag_path, path)
    append_to_sys_path(full_path)
    files.extend(get_python_files(full_path))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

import blender_node

base_class = blender_node.ObjectOps

# Import all the modules and append their mappings
for file in files:
    module = importlib.import_module(file)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if issubclass(obj, base_class) and obj != base_class:
                NODE_CLASS_MAPPINGS.update(obj.NODE_CLASS_MAPPINGS())
                NODE_DISPLAY_NAME_MAPPINGS.update(obj.NODE_DISPLAY_NAME_MAPPINGS())

    if hasattr(module, "BLENDER_NODES"):
        for node in module.BLENDER_NODES:
            # print(node)
            # NODE_CLASS_MAPPINGS.update({node: module.BLENDER_NODES[node]})
            # NODE_DISPLAY_NAME_MAPPINGS.update({node: node})
            NODE_CLASS_MAPPINGS.update(node.NODE_CLASS_MAPPINGS())
            NODE_DISPLAY_NAME_MAPPINGS.update(node.NODE_DISPLAY_NAME_MAPPINGS())

    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
