import os
import sys

os.environ["TYPE_GENERATION"] = "1"

current_dir = os.path.dirname(__file__)
blender_dir = os.path.join(current_dir, "blender")
sys.path.append(blender_dir)

import json
from blender.ops_mesh import BLENDER_NODES


with open(f"{blender_dir}/input_types.txt", "w") as f:
    for node in BLENDER_NODES:
        results = node.INPUT_TYPES()
        f.write(node.__name__ + "|" + json.dumps(results) + "\n")
