from aiohttp import web
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageOps
from dotenv import load_dotenv
import os
import requests
import folder_paths
import json
import numpy as np
import server
import re
import base64
from PIL import Image
import io
import time
import execution
import random

load_dotenv()


# For speeding up ONNX model, see https://github.com/facebookresearch/segment-anything/tree/main/demo#onnx-multithreading-with-sharedarraybuffer
def inject_headers(original_handler):
    async def _handler(request):
        res = await original_handler(request)
        res.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        res.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return res

    return _handler


routes = []
for item in server.PromptServer.instance.routes._items:
    if item.path == "/":
        item = web.RouteDef(
            method=item.method,
            path=item.path,
            handler=inject_headers(item.handler),
            kwargs=item.kwargs,
        )
    routes.append(item)
server.PromptServer.instance.routes._items = routes


@server.PromptServer.instance.routes.get("/avatar-graph-comfyui/tw-styles.css")
async def get_web_styles(request):
    filename = os.path.join(os.path.dirname(__file__), "js/tw-styles.css")
    return web.FileResponse(filename)


@server.PromptServer.instance.routes.get("/sam_model")
async def get_sam_model(request):
    model_type = request.rel_url.query.get("type", "vit_h")
    filename = os.path.join(folder_paths.base_path, f"web/models/sam_{model_type}.onnx")
    if not os.path.isfile(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Downloading ONNX model to {filename}")
        response = requests.get(
            f"https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/models/sam_{model_type}.onnx"
        )
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"ONNX model downloaded")
    return web.FileResponse(filename)


def load_image(image, is_generated_image):
    if is_generated_image:
        image_path = f"{folder_paths.get_output_directory()}/{image}"
    else:
        image_path = folder_paths.get_annotated_filepath(image)
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    return image


@server.PromptServer.instance.routes.post("/sam_model")
async def post_sam_model(request):
    post = await request.json()
    is_generated_image = post.get("isGeneratedImage")
    emb_id = post.get("embedding_id")
    ckpt = post.get("ckpt")
    ckpt = folder_paths.get_full_path("sams", ckpt)
    remote = post.get("remote")
    model_type = re.findall(r"vit_[lbh]", ckpt)[0]
    emb_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.npy"
    output_json_filename = (
        f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.json"
    )
    if not os.path.exists(emb_filename):
        image = load_image(post.get("image"), is_generated_image)
        if remote:
            # Run embed in remote server
            image = Image.fromarray((image * 255).astype(np.uint8))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image = base64.b64encode(buffered.getvalue()).decode()
            res = requests.post(
                "https://avatechgg--sam-embed.modal.run",
                headers={
                    "Content-type": "application/json",
                    "Accept": "application/json",
                },
                data=json.dumps(
                    {
                        "image": image,
                    }
                ),
            ).json()
            emb, input_size, original_size = (
                res["emb"],
                res["input_size"],
                res["original_size"],
            )
            emb = np.array(emb).astype(np.float32)
            np.save(emb_filename, emb)
            with open(output_json_filename, "w") as f:
                data = {
                    "input_size": input_size,
                    "original_size": original_size,
                }
                json.dump(data, f)
        else:
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            predictor = SamPredictor(sam)

            image_np = (image * 255).astype(np.uint8)
            predictor.set_image(image_np)
            emb = predictor.get_image_embedding().cpu().numpy()
            np.save(emb_filename, emb)
            with open(output_json_filename, "w") as f:
                json.dump(
                    {
                        "input_size": predictor.input_size,
                        "original_size": predictor.original_size,
                    },
                    f,
                )
    print("Finished embedding")
    return web.json_response({})


def save_image(image):
    input_folder = folder_paths.get_input_directory()
    name, extension = os.path.splitext(image.filename)
    save_name = f"{name}{extension}"
    i = 1
    while os.path.exists(f"{input_folder}/{save_name}"):
        save_name = f"{name}_{i}{extension}"
        i += 1

    with open(f"{input_folder}/{save_name}", "wb") as f:
        f.write(image.file.read())

    return save_name


def post_prompt(json_data):
    prompt_server = server.PromptServer.instance
    json_data = prompt_server.trigger_on_prompt(json_data)

    if "number" in json_data:
        number = float(json_data["number"])
    else:
        number = prompt_server.number
        if "front" in json_data:
            if json_data["front"]:
                number = -number

        prompt_server.number += 1

    if "prompt" in json_data:
        prompt = json_data["prompt"]
        valid = execution.validate_prompt(prompt)
        extra_data = {}
        if "extra_data" in json_data:
            extra_data = json_data["extra_data"]

        if "client_id" in json_data:
            extra_data["client_id"] = json_data["client_id"]
        if valid[0]:
            prompt_id = str(uuid.uuid4())
            outputs_to_execute = valid[2]
            prompt_server.prompt_queue.put(
                (number, prompt_id, prompt, extra_data, outputs_to_execute)
            )
            response = {
                "prompt_id": prompt_id,
                "number": number,
                "node_errors": valid[3],
            }
            return web.json_response(response)
        else:
            print("invalid prompt:", valid[1])
            return web.json_response(
                {"error": valid[1], "node_errors": valid[3]}, status=400
            )
    else:
        return web.json_response({"error": "no prompt", "node_errors": []}, status=400)


def get_avatar_file(outputs):
    for node_id, output in outputs.items():
        if "gltfFilename" in output:
            avatar_filename = output["gltfFilename"][0]
            with open(
                f"{folder_paths.get_output_directory()}/{avatar_filename}", "rb"
            ) as f:
                return f.read()


def upload_avatar_file(outputs):
    file = get_avatar_file(outputs)
    response = requests.get("https://labs.avatech.ai/api/share")
    labData = response.json()
    modelId = labData["modelId"]

    # upload model
    headers = {
        "x-amz-acl": "public-read",
        "Content-Type": "model/gltf-binary",
        "Content-Length": str(len(file)),
    }
    requests.put(labData["url"], headers=headers, data=file)

    # send notification
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    data = {
        "username": "Avabot",
        "avatar_url": "https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/avatechai.png",
        "content": "[API Call] New register!",
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))

    return modelId


def randomSeed(num_digits=15):
    range_start = 10 ** (num_digits - 1)
    range_end = (10**num_digits) - 1
    return random.randint(range_start, range_end)


with open(
    os.path.join(
        os.path.dirname(__file__),
        "workflow_templates/api/avatar_generation_mask_api.json",
    )
) as f:
    default_workflow = "\n".join(f.readlines())


@server.PromptServer.instance.routes.post("/avatar_generation")
async def post_prompt_block(request):
    prompt_server = server.PromptServer.instance
    post = await request.post()
    image = post.get("image")
    image_name = save_image(image)

    workflow = post.get("workflow")
    workflow = default_workflow if workflow is None else workflow
    api_prompt = json.loads(
        workflow.replace("IMAGE_REFERENCE", image_name).replace(
            "SEED", str(randomSeed())
        )
    )
    res = post_prompt({"prompt": api_prompt})
    prompt_id = json.loads(res.text)["prompt_id"]
    while True:
        history = prompt_server.prompt_queue.get_history(prompt_id=prompt_id)
        if history:
            # file = get_avatar_file(history[prompt_id]["outputs"])
            # return web.Response(body=file)

            modelId = upload_avatar_file(history[prompt_id]["outputs"])
            print("model id", modelId)
            return web.json_response({"id": modelId}, status=200)
        time.sleep(0.5)


# @server.PromptServer.instance.routes.get("/get_default_workflow")
# async def get_default_workflow(request):
#     # json_link = "https://cdn.discordapp.com/attachments/1119102674437156984/1172255632586448987/workflow_boy_2_1.json?ex=655fa722&is=654d3222&hm=463fa6a3c6ea60f7471196ff45382c729d3b856e86282f905d37a0398711860e&" # YP workflow
#     # json_link = "https://cdn.discordapp.com/attachments/729003657483518063/1172504658812608572/workflow_15.json?ex=65608f0e&is=654e1a0e&hm=f707d887b9294c1e9b26e54856b1e516d1725a1b25d044b46229cea6e5c804a1&" # Benny workflow
#     # json_link = 'https://cdn.discordapp.com/attachments/1110859802701221898/1173536418337914970/newstyle.json?ex=65644ff5&is=6551daf5&hm=f129838fae10197351bd27c69c7ff5eb4edf2c7d6ed74e6db8b55ddaa3c77dee&' # Deepwoo workflow
#     json_link = 'https://cdn.discordapp.com/attachments/729003657483518063/1174045115757633596/girl1114.json?ex=656629b8&is=6553b4b8&hm=df3d7798b887e2b3b6b06ea438f1bc4ba041dd0f9daf54ea48101845ec7f4243&'
#     response = requests.get(json_link)
#     response.raise_for_status()
#     return web.json_response(response.json())


@server.PromptServer.instance.routes.get("/get_workflow")
async def get_workflow(request):
    name = request.rel_url.query.get("name", "default")
    # if name == "default":
    #     json_link = 'https://cdn.discordapp.com/attachments/729003657483518063/1174045115757633596/girl1114.json?ex=656629b8&is=6553b4b8&hm=df3d7798b887e2b3b6b06ea438f1bc4ba041dd0f9daf54ea48101845ec7f4243&'
    #     response = requests.get(json_link)
    #     response.raise_for_status()
    #     workflow = response.json()
    # else:
    if name == "default":
        name = "Auto_segment_workflow"

    workflows_path = os.path.join(os.path.dirname(__file__), "workflow_templates")
    workflow = json.load(open(f"{workflows_path}/{name}.json"))
    return web.json_response(workflow)


@server.PromptServer.instance.routes.post("/segments")
async def post_segments(request):
    post = await request.json()
    name = post.get("name")
    segments = post.get("segments")
    output_dir = os.path.join(folder_paths.base_path, f"output/segments_{name}")
    os.makedirs(output_dir, exist_ok=True)
    for key, value in segments.items():
        filename = os.path.join(output_dir, f"{key}.png")
        with open(filename, "wb") as f:
            f.write(base64.b64decode(value.split(",")[1]))

    order = list(segments.keys())
    with open(os.path.join(output_dir, "order.json"), "w") as f:
        json.dump(order, f)
    return web.json_response({})


# @server.PromptServer.instance.routes.post("/segments_order")
# async def post_segments(request):
#     post = await request.json()
#     name = post.get("name")
#     order = post.get("order")
#     output_dir = os.path.join(folder_paths.base_path, f"output/{name}")
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "order.json") , "w") as f:
#         json.dump(order, f)
#     return web.json_response({})


@server.PromptServer.instance.routes.get("/get_webhook")
async def get_webhook(request):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    return web.json_response(url)


import uuid


@server.PromptServer.instance.routes.post("/create_avatar_from_image")
async def post_input_file(request):
    post = await request.read()

    # Doesn't seems working when file isnt png / or nothing is uploaded
    if not post:
        raise web.HTTPBadRequest(reason="No image data received")

    try:
        queue_id = uuid.uuid4()

        output_dir = os.path.join(
            folder_paths.base_path, "input", "create_avatar_endpoint"
        )
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, str(queue_id) + ".png")
        with open(filename, "wb") as f:
            f.write(post)

        return web.json_response(
            {
                "redirect_url": "https://ai-assistant.avatech.ai?queue-id="
                + str(queue_id)
            }
        )
    except Exception as e:
        print(e)
        return web.json_response({"error": e})
