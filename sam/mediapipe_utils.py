# For auto-segmentation
import mediapipe as mp
import numpy as np
import os
from math import sqrt

face_landmarker = None
pose_landmarker = None

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": -0.3,
        "negativeScale": 0.3,
        # https://stackoverflow.com/questions/66649492/how-to-get-specific-landmark-of-face-like-lips-or-eyes-using-tensorflow-js-face
        "indices": [[x, x] for x in [61, 37, 270, 91, 314]],
    },
    "mouth_in": {
        "useMiddle": False,
        "positiveOffsetX": 0,
        "positiveOffsetY": 0,
        "negativeOffsetX": 0,
        "negativeOffsetY": 0,
        "positiveScale": -0.5,
        "negativeScale": 0.5,
        # https://stackoverflow.com/questions/66649492/how-to-get-specific-landmark-of-face-like-lips-or-eyes-using-tensorflow-js-face
        "indices": [[x, x] for x in [310, 88]],
    },
}


def load_mediapipe_models():
    global face_landmarker, pose_landmarker
    if face_landmarker is None and pose_landmarker is None:
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
    return face_landmarker, pose_landmarker


def auto_segment(image, face_landmarks, pose_landmarks):
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

                positivePointDistance = value["positiveScale"] * directionVectorLength
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
            positiveBreathX = ((pose_landmarks[11].x + pose_landmarks[12].x) / 2) * W
            positiveBreathY = ((pose_landmarks[11].y + pose_landmarks[12].y) / 2) * H
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


def detect_face(np_image):
    face_landmarker, pose_landmarker = load_mediapipe_models()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=(np_image * 255).astype(np.uint8)
    )
    face_landmarks = face_landmarker.detect(mp_image).face_landmarks
    if len(face_landmarks) == 0:
        print("Warning: no face detected")
        return None, None

    pose_landmarks = pose_landmarker.detect(mp_image).pose_landmarks
    if len(pose_landmarks) == 0:
        print("Warning: no pose detected")
        return None, None

    imagePromptsMulti, boxesMulti = auto_segment(
        np_image, face_landmarks[0], pose_landmarks[0]
    )

    return imagePromptsMulti, boxesMulti
