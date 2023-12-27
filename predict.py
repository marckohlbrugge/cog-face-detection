#!/usr/bin/env python3

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import mediapipe as mp
from PIL import Image
import numpy as np
import zipfile
import os

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.1
        )

    def predict(
        self,
        image: Path = Input(
            description="Path to image file or zip file",
        ),
    ) -> dict:
        """Run a single prediction on the model"""
        if zipfile.is_zipfile(image):
            face_coordinates = {}
            with zipfile.ZipFile(image, 'r') as zip_ref:
                zip_ref.extractall("images")
            images = os.listdir("images")
            for img in images:
                image = Image.open(os.path.join("images", img))
                image_np = np.array(image)
                image_width, image_height = image.size

                # Perform face detection
                results_detection = self.face_detection.process(image_np)
                if results_detection.detections:
                    face_coordinates[img] = []
                    for detection in results_detection.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * image_width)
                        y = int(bbox.ymin * image_height)
                        width = int(bbox.width * image_width)
                        height = int(bbox.height * image_height)
                        face_coordinates[img].append({"x": x, "y": y, "width": width, "height": height})
                else:
                        face_coordinates[img] = []
        else:
            face_coordinates = {"faces": []}
            image = Image.open(image)
            image_np = np.array(image)
            image_width, image_height = image.size

            # Perform face detection
            results_detection = self.face_detection.process(image_np)
            if results_detection.detections:
                for detection in results_detection.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * image_width)
                    y = int(bbox.ymin * image_height)
                    width = int(bbox.width * image_width)
                    height = int(bbox.height * image_height)
                    face_coordinates["faces"].append({"x": x, "y": y, "width": width, "height": height})

        return face_coordinates


