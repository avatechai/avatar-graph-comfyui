# avatar-graph-comfyui

![image](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/201a005b-7e00-4671-85a1-54937bf0704e)



https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/b229a4b3-88ba-4755-a8e0-9149260eef12




This introduce a custom workflow on top of comfyui built for avatar-graph.

# Custom Nodes

| Category | Node Name | Description |
| --- | --- | --- |
| Segmentation | Segmentation (SAM) |  |
| Mesh | Create Mesh Layer | Create a mesh object from the input images (usually a segmented part of the entire image) |
| Blendshapes | Mesh Modify Shape Key | Given shape key name & target vertex_group, modify the vertex / all vertex’s transform |
|  | Create Shape Flow  | Create runtime shape flow graph, allowing interactive inputs affecting shape keys value in runtime |
| UV & Texture | Match Texture Aspect Ratio | Since the mesh is created in 1:1 aspect ratio, a re-scale is needed at the end of the operation |
| Output | Avatar Main Output |  |

# Workflow

# Shape Flow
![image](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/e95aa89c-5c1d-4ef4-89e9-37aa8ad8bb55)


# Installation

1. `cd custom_nodes`

2. `git clone https://github.com/avatechgg/avatar-graph-comfyui.git`

3. Install deps `cd avatar-graph-comfyui && python -m pip install -r requirements.txt`

4. Navigate to in your comfyui installation folder, open `comfyui/main.py` and add the following lines:
```py
def prompt_worker(q, server):
    e = execution.PromptExecutor(server)

    # This is a workaround since using blender as a python module, the bpy module has to be imported after in the custom thread, otherwise it will cause a segfault if imported in the custom nodes.

    # Add next line
    import global_bpy

    while True:
        item, item_id = q.get()
        # Add next line
        global_bpy.reset_bpy()

if __name__ == "__main__":
    # Add next two lines 
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "custom_nodes", "avatar-graph-comfyui"))


```

5. Restart comfyui

6. Run comfyui with enable-cors-header `python main.py --enable-cors-header` or (mac)`python main.py --force-fp16 --enable-cors-header`

# Development

For comfyui frontend extension, frontend js located at `avatar-graph-comfyui/js`

Web stack used: [vanjs](https://github.com/vanjs-org/van) [tailwindcss](https://github.com/tailwindlabs/tailwindcss)

## Install deps

```
pnpm i
```

Run the dev command to start the tailwindcss watcher

```
pnpm dev
```

For each changes, simply refresh the comfyui page to see the changes.

p.s. For tailwind autocomplete, add the following to your vscode settings.json.

```json
{
    "tailwindCSS.experimental.classRegex": [
        ["class\\s?:\\s?([\\s\\S]*)", "(?:\"|')([^\"']*)(?:\"|')"]
    ]
}
```
