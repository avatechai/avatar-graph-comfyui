# avatar-graph-comfyui

This introduce a custom workflow on top of comfyui built for avatar-graph.

# Installation

1. `cd custom_nodes`

2. `git clone https://github.com/avatechgg/avatar-graph-comfyui.git`

3. Install deps `python -m pip install -r requirements.txt`

4. Navigate to in your comfyui installation folder, open `comfyui/main.py` and add the following lines:
```py
def prompt_worker(q, server):
    e = execution.PromptExecutor(server)

    # This is a workaround since using blender as a python module, the bpy module has to be imported after in the custom thread, otherwise it will cause a segfault if imported in the custom nodes.

    # Add this line here
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "custom_nodes", "avatar-graph-comfyui"))
    import global_bpy

    while True:
        item, item_id = q.get()
        # Add this line here
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

```
{
    "tailwindCSS.experimental.classRegex": [
        ["class\\s?:\\s?([\\s\\S]*)", "(?:\"|')([^\"']*)(?:\"|')"]
    ]
}
```