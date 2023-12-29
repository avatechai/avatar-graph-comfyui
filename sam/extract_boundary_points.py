import cv2
import numpy as np
import torch


class ExtractBoundaryPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "n_points": ("INT", {"default": -1, "min": -1, "max": 100}),
            },
        }

    RETURN_TYPES = ("POINTS", "IMAGE")

    FUNCTION = "run"

    CATEGORY = "image"

    def find_main_contour(self, image, n_points):
        image = np.copy(image[0].numpy())
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = (gray * 255).astype(np.uint8)
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise Exception(
                "No contours found. Please ensure that the image has the correct segments (e.g. when you click on the mouth, it should display a proper blue area over the mouth region)."
            )

        # Get the largest contour
        areas = [cv2.contourArea(contour) for contour in contours]

        max_area_index = areas.index(max(areas))
        largest_contour = contours[max_area_index]
        if n_points > 0:
            divided_by = int(largest_contour.shape[0] / n_points)
            divided_by = min(divided_by, largest_contour.shape[0])
            largest_contour = largest_contour[::divided_by].astype(int)
        contours = [largest_contour]

        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        points = []
        for point in largest_contour:
            points.append(
                {"x": point[0][0], "y": point[0][1], "label": 1, "isAuto": True}
            )

        return image, points

    def run(self, image, n_points):
        contour_image, points = self.find_main_contour(image, n_points)
        contour_image = torch.from_numpy(np.expand_dims(contour_image, axis=0))
        print(points)
        return (points, contour_image)


NODE_CLASS_MAPPINGS = {"Extract Boundary Points": ExtractBoundaryPoints}

NODE_DISPLAY_NAME_MAPPINGS = {"Extract Boundary Points": "Extract Boundary Points"}
