import folder_paths
import os
import numpy as np
import torch
import re
import json
from segment_anything import sam_model_registry, SamPredictor
from einops import rearrange, repeat
from PIL import Image
import mediapipe as mp
from math import sqrt

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

global_predictor = None
face_landmarker = None
pose_landmarker = None

# For auto-segmentation
layerMapping = {
    "L_eye": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": 0,
        "negativeScale": 0.5,
        "indices": mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
    },
    "R_eye": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": 0,
        "negativeScale": 0.5,
        "indices": mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
    },
    "L_iris": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": -0.2,
        "negativeScale": 0.5,
        "indices": mp.solutions.face_mesh.FACEMESH_LEFT_IRIS,
    },
    "R_iris": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": -0.2,
        "negativeScale": 0.5,
        "indices": mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS,
    },
    "face": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 40,
        "negativeOffsetX": 0,
        "negativeOffsetY": 60,
        "positiveScale": 0.2,
        "negativeScale": 0.6,
        "indices": mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
    },
    "mouth": {
        "useMiddle": True,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": 0,
        "negativeScale": 0,
        "indices": mp.solutions.face_mesh.FACEMESH_LIPS,
    },
    "mouth_in": {
        "useMiddle": True,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": 0,
        "negativeScale": 0,
        "indices": mp.solutions.face_mesh.FACEMESH_LIPS,
    },
}


class SAMMultiLayer:
    def __init__(self):
        self.predictor = None
        self.output_dir = folder_paths.get_output_directory()

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

    RETURN_TYPES = ("SAM_PROMPT",)
    FUNCTION = "load_image"

    def load_models(self, ckpt, model_type):
        global global_predictor, face_landmarker, pose_landmarker

        ckpt = folder_paths.get_full_path("sams", ckpt)
        sam = sam_model_registry[model_type](checkpoint=ckpt)  # .to("cuda")
        global_predictor = SamPredictor(sam)

        face_landmarker_model_path = os.path.join(
            os.path.dirname(__file__), "../mediapipe_models/face_landmarker.task"
        )
        face_landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_landmarker_model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)

        pose_landmarker_model_path = os.path.join(
            os.path.dirname(__file__), "../mediapipe_models/pose_landmarker_full.task"
        )
        pose_landmarker_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_landmarker_model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_landmarker_options)
        return global_predictor, face_landmarker, pose_landmarker

    def auto_segment(self, image, face_landmarks, pose_landmarks):
        H, W, C = image.shape
        imagePromptsMulti = {}
        boxesMulti = {}

        for key, value in layerMapping.items():
            positivePoints = []
            middlePoints = []
            negativePoints = []

            for index in value["indices"]:
                start, end = index
                startPoint = face_landmarks[start]

                startX = startPoint.x * W
                startY = startPoint.y * H

                if len(middlePoints) == 0:
                    middlePoints.append({"x": startX, "y": startY, "label": 1})
                else:
                    middlePoints[0]["x"] += startX
                    middlePoints[0]["y"] += startY

                positivePoints.append({"x": startX, "y": startY, "label": 1})

            len_indices = len(value["indices"])
            middlePoints[0]["x"] /= len_indices
            middlePoints[0]["y"] /= len_indices

            if value["useMiddle"]:
                imagePromptsMulti[key] = middlePoints
            else:
                for i, index in enumerate(value["indices"]):
                    start, end = index
                    startPoint = face_landmarks[start]

                    startX = startPoint.x * W
                    startY = startPoint.y * H

                    middlePoint = middlePoints[0]
                    directionVector = {
                        "x": middlePoint["x"] - startX,
                        "y": middlePoint["y"] - startY,
                    }
                    directionVectorLength = sqrt(
                        directionVector["x"] * directionVector["x"]
                        + directionVector["y"] * directionVector["y"]
                    )

                    if value["negativeScale"] != 0:
                        negativePointDistance = (
                            value["negativeScale"] * directionVectorLength
                        )
                        negativePoint = {
                            "x": startX
                            - (negativePointDistance * directionVector["x"])
                            / directionVectorLength
                            - value["negativeOffsetX"],
                            "y": startY
                            - (negativePointDistance * directionVector["y"])
                            / directionVectorLength
                            - value["negativeOffsetY"],
                            "label": 0,
                        }
                        negativePoints.append(negativePoint)

                    positivePointDistance = (
                        value["positiveScale"] * directionVectorLength
                    )
                    positivePoints[i] = {
                        "x": positivePoints[i]["x"]
                        - (positivePointDistance * directionVector["x"])
                        / directionVectorLength
                        - value["positiveOffsetX"],
                        "y": positivePoints[i]["y"]
                        - (positivePointDistance * directionVector["y"])
                        / directionVectorLength
                        - value["positiveOffsetY"],
                        "label": 1,
                    }

                imagePromptsMulti[key] = positivePoints + negativePoints

            points = negativePoints if len(negativePoints) > 0 else positivePoints
            box = np.array(
                [
                    min(x["x"] for x in points),
                    min(x["y"] for x in points),
                    max(x["x"] for x in points),
                    max(x["y"] for x in points),
                ]
            )
            boxesMulti[key] = box

            if pose_landmarks is not None:
                positiveBreathX = (
                    (pose_landmarks[11].x + pose_landmarks[12].x) / 2
                ) * W
                positiveBreathY = (
                    (pose_landmarks[11].y + pose_landmarks[12].y) / 2
                ) * H
                negativeBreathX1 = pose_landmarks[0].x * W
                negativeBreathY1 = pose_landmarks[0].y * H
                negativeBreathX2 = pose_landmarks[9].x * W
                negativeBreathY2 = pose_landmarks[9].y * H
                negativeBreathX3 = pose_landmarks[10].x * W
                negativeBreathY3 = pose_landmarks[10].y * H
                imagePromptsMulti["breath"] = [
                    {"x": positiveBreathX, "y": positiveBreathY, "label": 1},
                    {"x": negativeBreathX1, "y": negativeBreathY1, "label": 0},
                    {"x": negativeBreathX2, "y": negativeBreathY2, "label": 0},
                    {"x": negativeBreathX3, "y": negativeBreathY3, "label": 0},
                ]

        return imagePromptsMulti, boxesMulti

    def detect_face(self, np_image):
        global face_landmarker, pose_landmarker
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=(np_image * 255).astype(np.uint8)
        )
        face_landmarks = face_landmarker.detect(mp_image).face_landmarks
        face_landmarks = face_landmarks[0] if len(face_landmarks) > 0 else None
        pose_landmarks = pose_landmarker.detect(mp_image).pose_landmarks
        pose_landmarks = pose_landmarks[0] if len(pose_landmarks) > 0 else None
        imagePromptsMulti, boxesMulti = self.auto_segment(
            np_image, face_landmarks, pose_landmarks
        )

        return imagePromptsMulti, boxesMulti

    def load_image(self, image, ckpt, embedding_id, image_prompts_json):
        image_prompts = json.loads(image_prompts_json)

        order_file = f"{self.output_dir}/segments_{embedding_id}/order.json"
        if os.path.exists(order_file):
            # Frontend uploads segments images to backend => backend reads all segments images and passes them to next nodes
            with open(order_file) as f:
                order = json.load(f)

            result = [image_prompts]

            for segment in order:
                image = Image.open(
                    f"{self.output_dir}/segments_{embedding_id}/{segment}.png"
                )
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                result.append(image)

            return result
        else:
            # Frontend uploads clicks coordinates to backend => backend runs SAM and passes the segments to next nodes
            model_type = re.findall(r"vit_[lbh]", ckpt)[0]

            global global_predictor
            if global_predictor is None:
                global_predictor, _, _ = self.load_models(ckpt, model_type)

            if image.shape[3] == 4:
                image = image[:, :, :, :3]

            emb_filename = f"{self.output_dir}/{embedding_id}_{model_type}.npy"
            if not os.path.exists(emb_filename):
                image_np = (image[0].numpy() * 255).astype(np.uint8)
                global_predictor.set_image(image_np)
                emb = global_predictor.get_image_embedding().cpu().numpy()
                np.save(emb_filename, emb)

                with open(
                    f"{self.output_dir}/{embedding_id}_{model_type}.json", "w"
                ) as f:
                    data = {
                        "input_size": global_predictor.input_size,
                        "original_size": global_predictor.original_size,
                    }
                    json.dump(data, f)
            else:
                emb = np.load(emb_filename)

                with open(f"{self.output_dir}/{embedding_id}_{model_type}.json") as f:
                    data = json.load(f)
                    global_predictor.input_size = data["input_size"]
                    global_predictor.features = torch.from_numpy(emb)
                    global_predictor.is_image_set = True
                    global_predictor.original_size = data["original_size"]

            imagePromptsMulti, boxesMulti = self.detect_face(image[0].numpy())

            image_prompts = json.loads(image_prompts_json)
            result = [image_prompts]

            if isinstance(image_prompts, list):
                pass
            elif all(isinstance(item, list) for item in image_prompts.values()):
                for key, item in image_prompts.items():
                    if len(item) == 0:
                        h, w, c = image[0].shape
                        result.append(torch.zeros(1, h, w, c))
                        continue

                    points = (
                        item + imagePromptsMulti[key]
                        if key in imagePromptsMulti
                        else item
                    )
                    point_coords = np.array([[p["x"], p["y"]] for p in points])
                    point_labels = np.array([p["label"] for p in points])

                    masks, _, _ = global_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=boxesMulti[key] if key in boxesMulti else None,
                    )
                    masks = torch.from_numpy(masks)
                    masks = rearrange(masks[0], "h w -> 1 h w")
                    out_image = repeat(masks, "1 h w -> 1 h w c", c=3) * image
                    result.append(out_image)
            return result


NODE_CLASS_MAPPINGS = {"SAM MultiLayer": SAMMultiLayer}

NODE_DISPLAY_NAME_MAPPINGS = {"SAM MultiLayer": "SAM MultiLayer"}
