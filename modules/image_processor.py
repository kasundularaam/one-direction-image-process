from PIL import Image
from ultralytics import YOLO
import numpy as np
from .models import ArrowDetectionResult


class YOLOImageProcessor:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)
        self.class_labels = ['left', 'right', 'forward']

    async def process_image(self, image: Image.Image) -> tuple[ArrowDetectionResult, Image.Image]:
        """
        Process an image with YOLO model and return detection results and annotated image.

        Args:
            image (PIL.Image): Input image to process

        Returns:
            tuple: (ArrowDetectionResult, PIL.Image) - Detection results and annotated image
        """
        # Run inference
        results = self.model(image)
        result = results[0]  # Get first result

        # Get boxes from results
        boxes = result.boxes

        # Create a list of detections
        detections = []
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append({
                'confidence': conf,
                'class': cls,
                'bbox': xyxy
            })

        # Sort detections by confidence
        sorted_detections = sorted(
            detections, key=lambda x: x['confidence'], reverse=True)

        # Get plotted image with annotations
        plotted_image = result.plot()
        plotted_image_pil = Image.fromarray(
            plotted_image[..., ::-1])  # BGR to RGB

        # Create result object
        result_data = ArrowDetectionResult(
            arrow_count=min(len(sorted_detections), 3)
        )

        # Fill in the top 3 detections
        for i, detection in enumerate(sorted_detections[:3]):
            bbox = detection['bbox']
            label = self.class_labels[detection['class']]
            confidence = detection['confidence']

            # Set attributes dynamically
            setattr(result_data, f'arrow_label{i+1}', label)
            setattr(result_data, f'confidence{i+1}', float(confidence))
            setattr(result_data, f'x_min{i+1}', float(bbox[0]))
            setattr(result_data, f'y_min{i+1}', float(bbox[1]))
            setattr(result_data, f'x_max{i+1}', float(bbox[2]))
            setattr(result_data, f'y_max{i+1}', float(bbox[3]))

        return result_data, plotted_image_pil
