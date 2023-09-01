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

    # Add this line here
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "custom_nodes", "avatar-graph-comfyui"))
    import global_bpy

    while True:
        item, item_id = q.get()
        # Add this line here
        global_bpy.reset_bpy()

```

4. Restart comfyui