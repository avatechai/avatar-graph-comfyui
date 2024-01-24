import folder_paths
import os
import numpy as np
import torch
import re
import json
import uuid
from sam_utils import (
    load_model,
    check_embedding_exists,
    compute_image_embedding,
    save_embedding,
    load_embdding,
    load_image,
)
from mediapipe_utils import detect_face
from einops import rearrange, repeat


class SAMMultiLayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ckpt": (folder_paths.get_filename_list("sams"),),
                "embedding_id": (
                    "STRING",
                    {"multiline": False, "default": "embedding"},
                ),
                "image_prompts_json": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ["SAM_PROMPT"] # + ["IMAGE"] * 100
    FUNCTION = "run"

    def run(self, image, ckpt, embedding_id, image_prompts_json):
        if (
            "COMFY_DEPLOY" in os.environ
            and os.getenv("COMFY_DEPLOY", "FALSE") == "TRUE"
        ):
            embedding_id = str(uuid.uuid4())
        layer_points = json.loads(image_prompts_json.replace("'", '"'))

        order_file = (
            f"{folder_paths.get_output_directory()}/segments_{embedding_id}/order.json"
        )
        if os.path.exists(order_file):
            # Frontend uploads segments images to backend => backend reads all segments images and passes them to next nodes
            with open(order_file) as f:
                order = json.load(f)

            result = [layer_points]
            for segment in order:
                image = load_image(
                    f"{folder_paths.get_output_directory()}/segments_{embedding_id}/{segment}.png",
                    comfyui_format=True,
                )
                result.append(image)

            return result
        else:
            # Frontend uploads clicks coordinates to backend => backend runs SAM and passes the segments to next nodes
            model_type = re.findall(r"vit_[lbh]", ckpt)[0]

            # get first image from batch
            image = image[0]
            if image.shape[2] == 4:
                # to RGB
                image = image[:, :, :3]

            if not check_embedding_exists(embedding_id, model_type):
                emb, img_model_input_size, img_original_size = compute_image_embedding(
                    image, model_type
                )
                save_embedding(
                    embedding_id,
                    model_type,
                    emb,
                    img_model_input_size,
                    img_original_size,
                )
            else:
                load_embdding(embedding_id, model_type)

            detected_points, detected_bboxes = detect_face(image.numpy())
            result = [layer_points]
            for layer, points in layer_points.items():
                if detected_points is not None and layer in detected_points:
                    # use detected points by mediapipe
                    points = detected_points[layer]

                if len(points) == 0:
                    # no points, append a black image
                    h, w, c = image.shape
                    result.append(torch.zeros(1, h, w, c))
                    print("No points for layer", layer)
                    continue

                # prepare for SAM inferencing
                point_coords = np.array([[p["x"], p["y"]] for p in points])
                point_labels = np.array([p["label"] for p in points])
                bbox = (
                    detected_bboxes[layer]
                    if detected_bboxes is not None and layer in detected_bboxes
                    else None
                )

                sam_predictor = load_model(model_type)["predictor"]
                masks, _, _ = sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=bbox,
                )
                masks = torch.from_numpy(masks)
                masks = rearrange(masks[0], "h w -> 1 h w")
                out_image = repeat(masks, "1 h w -> 1 h w c", c=3) * image.unsqueeze(0)
                result.append(out_image)
        return result


NODE_CLASS_MAPPINGS = {"SAM MultiLayer": SAMMultiLayer}

NODE_DISPLAY_NAME_MAPPINGS = {"SAM MultiLayer": "SAM MultiLayer"}
