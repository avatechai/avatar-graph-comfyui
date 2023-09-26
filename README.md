# avatar-graph-comfyui

![image](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/201a005b-7e00-4671-85a1-54937bf0704e)

A custom_nodes module for creating real-time interactive avatars powered by blender bpy mesh api + Avatech Shape Flow runtime.

# Demo

https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/b229a4b3-88ba-4755-a8e0-9149260eef12

# Custom Nodes

| Preview                                                                                                                                     | Node Name                  | Description                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------- |
| ![CleanShot 2023-09-26 at 15 42 46](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/8aabeba8-5450-4d39-8203-e91f9ab47190) | Segmentation (SAM)         | Integrative SAM node allowing you to directly select and create multiple image segment output.        |
| ![CleanShot 2023-09-26 at 15 44 01](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/40740d25-9411-4cd3-a6c0-8b9008bca41c) | Create Mesh Layer          | Create a mesh object from the input images (usually a segmented part of the entire image)             |
| ![CleanShot 2023-09-26 at 15 44 29](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/ba7afbc5-9cd5-4f97-9614-f71133f5783e) | Join Meshes                | Combine multiple meshes into a single mesh object                                                     |
| ![CleanShot 2023-09-26 at 15 48 26](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/ab4f259c-89a7-4f51-bc54-fd179e252073) | Mesh Modify Shape Key      | Given shape key name & target vertex_group, modify the vertex / all vertex’s transform                |
| ![CleanShot 2023-09-26 at 16 38 51](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/abfdd801-0387-4c5d-9c11-6c23337ff1dd) | Create Shape Flow          | Create runtime shape flow graph, allowing interactive inputs affecting shape keys value in runtime    |
| ![CleanShot 2023-09-26 at 17 01 51](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/cb7155be-fb31-49f8-a24a-d001a1484ea7) | Match Texture Aspect Ratio | Since the mesh is created in 1:1 aspect ratio, a re-scale is needed at the end of the operation       |
| ![CleanShot 2023-09-26 at 17 11 44](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/4b9c0cf5-0497-47bf-8e06-5a3370084c11) | Plane Texture Unwrap       | Will perform mesh face fill and UV Cube project on the target plane mesh, scaled to bounds.           |
| ![CleanShot 2023-09-26 at 16 37 54](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/6a9a8bb4-05ec-4a2e-98bf-194b6af3a62a) | Avatar Main Output         | The primary output of the .ava file. The embeded Avatar View will auto update with this node's output |

# Workflow

<details>
<summary>  Workflow01 </summary>
![eye+mouth movement](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/8a237b9d-05fc-4e4a-b802-6465911f0d77)


➡️[Workflow](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/SimpleEye+MouthMovement.json)
</details>

<details>
<summary> Character Gen Prompting Guide </summary>
    
### Notice: 
- We need a character image with an open mouth and enable the tool to easily recognize facial features, so please add to the prompt: ```looking at viewer, detailed face, open mouth, [smile], solo,eye-level angle```

### Workflow Download:

➡️[Simple CharacterGen Workflow](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/SimpleCharacterGen.json)
- Feel free to change any checkpoint model that suits your needs.
  
![image](https://github.com/avatechai/avatar-graph-comfyui/assets/48451938/acea9933-359b-4398-8d2a-582bf02bef99)

  
</details>

<details>
<summary>  Mouth Open(Inpaint) Guide </summary>
    
### Notice: 
- To maintain consistency with the base image, it is recommended to utilize a checkpoint model that aligns with its style.

### Workflow Download:

➡️[MouthOpen Workflow](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/MouthOpen_(inpaint).json)

![inpaint_workflow](https://github.com/avatechai/avatar-graph-comfyui/assets/48451938/d11d840b-7ea6-4b47-bc26-a2af7c8c27a5)

### Inpaint Demonstration 

https://github.com/avatechai/avatar-graph-comfyui/assets/48451938/ff48c3d9-7292-4505-8993-8f117cee34ff

### Recommend Checkpoint Model List:
##### Anime Style SD1.5
- https://civitai.com/models/35960/flat-2d-animerge
- https://civitai.com/models/24149/mistoonanime
- https://civitai.com/models/22364/kizuki-anime-hentai-checkpoint
##### Realistic Style SD1.5
- https://civitai.com/models/4201/realistic-vision-v51
- https://civitai.com/models/49463/am-i-real
- https://civitai.com/models/43331/majicmix-realistic
- 
</details>

# Shape Flow
![image](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/a834e535-4f87-4b77-81a6-435e3a67ca4a)

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
