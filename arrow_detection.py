from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ArrowDetectionResult:
    arrow_count: int
    # Arrow 1
    arrow_label1: Optional[str] = None
    confidence1: Optional[float] = None
    x_min1: Optional[float] = None
    y_min1: Optional[float] = None
    x_max1: Optional[float] = None
    y_max1: Optional[float] = None
    # Arrow 2
    arrow_label2: Optional[str] = None
    confidence2: Optional[float] = None
    x_min2: Optional[float] = None
    y_min2: Optional[float] = None
    x_max2: Optional[float] = None
    y_max2: Optional[float] = None
    # Arrow 3
    arrow_label3: Optional[str] = None
    confidence3: Optional[float] = None
    x_min3: Optional[float] = None
    y_min3: Optional[float] = None
    x_max3: Optional[float] = None
    y_max3: Optional[float] = None
    # Direction
    direction: str = "stop"


async def process_image_with_yolo(image: Image.Image) -> ArrowDetectionResult:
    # Load the YOLO model
    model = YOLO("best.pt")

    # Run inference
    results = model(image)
    # Get the first result since we're processing a single image
    result = results[0]

    # Get boxes from results
    boxes = result.boxes

    # Create a list of detections with confidence and coordinates
    detections = []
    for box in boxes:
        # Get confidence
        conf = float(box.conf[0])
        # Get class name
        cls = int(box.cls[0])
        # Get coordinates (already in x1,y1,x2,y2 format)
        xyxy = box.xyxy[0].cpu().numpy()
        detections.append({
            'confidence': conf,
            'class': cls,
            'bbox': xyxy
        })

    # Sort detections by confidence
    sorted_detections = sorted(
        detections, key=lambda x: x['confidence'], reverse=True)

    # Initialize result
    result = ArrowDetectionResult(
        arrow_count=min(len(sorted_detections), 3)
    )

    # Map class indices to labels
    class_labels = ['left', 'right', 'forward']

    # Fill in the top 3 detections (or fewer if less than 3 detected)
    for i, detection in enumerate(sorted_detections[:3]):
        bbox = detection['bbox']
        label = class_labels[detection['class']]
        confidence = detection['confidence']

        # Set attributes dynamically based on detection index
        setattr(result, f'arrow_label{i+1}', label)
        setattr(result, f'confidence{i+1}', float(confidence))
        setattr(result, f'x_min{i+1}', float(bbox[0]))
        setattr(result, f'y_min{i+1}', float(bbox[1]))
        setattr(result, f'x_max{i+1}', float(bbox[2]))
        setattr(result, f'y_max{i+1}', float(bbox[3]))

    return result
