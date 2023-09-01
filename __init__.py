"""
@author: Avatech Limited
@title: Avatar Graph
@nickname: Avatar Graph
@description: Include nodes for sam + bpy operation, that allows workflow creations for generative 2d character rig.
"""

import os
import sys
import importlib
import subprocess

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
    return [f[:-3] for f in os.listdir(path) if f.endswith('.py')]

def append_to_sys_path(path):
    if path not in sys.path:
        sys.path.append(path)

paths = ['blender', 'sam', 'common']
files = []

for path in paths:
    full_path = os.path.join(ag_path, path)
    append_to_sys_path(full_path)
    files.extend(get_python_files(full_path))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all the modules and append their mappings
for file in files:
    module = importlib.import_module(file)
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']