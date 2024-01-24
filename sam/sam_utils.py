from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageOps
import folder_paths
import numpy as np
import torch
import os
import json

sam_type_to_ckpt = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
sam_ckpt_to_type = {v: k for k, v in sam_type_to_ckpt.items()}

sam_instance = {"model_type": None, "model": None, "predictor": None}


def load_model(model_type):
    global sam_instance
    if sam_instance["model"] is None or sam_instance["model_type"] != model_type:
        ckpt = sam_type_to_ckpt[model_type]
        ckpt = folder_paths.get_full_path("sams", ckpt)
        sam_instance["model_type"] = model_type
        sam_instance["model"] = sam_model_registry[model_type](checkpoint=ckpt)
        if torch.cuda.is_available():
            sam_instance["model"].cuda()
        sam_instance["predictor"] = SamPredictor(sam_instance["model"])
    return sam_instance


def check_embedding_exists(emb_id, model_type):
    emb_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.npy"
    return os.path.exists(emb_filename)


def save_embedding(emb_id, model_type, emb, img_input_size, img_original_size):
    emb_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.npy"
    np.save(emb_filename, emb)

    json_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.json"
    with open(json_filename, "w") as f:
        json.dump(
            {
                "input_size": img_input_size,
                "original_size": img_original_size,
            },
            f,
        )


def load_embdding(emb_id, model_type):
    emb_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.npy"
    emb = np.load(emb_filename)

    json_filename = f"{folder_paths.get_output_directory()}/{emb_id}_{model_type}.json"
    with open(json_filename, "r") as f:
        sizes = json.load(f)

    predictor = load_model(model_type)["predictor"]
    predictor.input_size = sizes["input_size"]
    predictor.features = torch.from_numpy(emb)
    predictor.is_image_set = True
    predictor.original_size = sizes["original_size"]


@torch.no_grad()
def compute_image_embedding(image, model_type="vit_h"):
    sam = load_model(model_type)
    predictor = sam["predictor"]

    # if image.shape[3] == 4:
    #     image = image[:, :, :, :3]

    image_np = (image * 255).astype(np.uint8)
    predictor.set_image(image_np)
    emb = predictor.get_image_embedding().cpu().numpy()

    return emb, predictor.input_size, predictor.original_size


def load_image(image, is_generated_image=False, comfyui_format=False):
    if is_generated_image:
        image_path = f"{folder_paths.get_output_directory()}/{image}"
    else:
        image_path = folder_paths.get_annotated_filepath(image)
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    if comfyui_format:
        # to torch and create batch dimension
        image = torch.from_numpy(image)[None,]
    return image
