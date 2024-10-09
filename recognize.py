from imageai.Detection import ObjectDetection
import os
import warnings
import io
import numpy as np

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="torch.serialization")

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
model_path = os.path.join(execution_path, "yolov3.pt")

# Load the model
detector.setModelPath(model_path)
detector.loadModel()


def find_objects(image_bytes):
    # Save the incoming image bytes to a temporary file
    temp_input_path = os.path.join(execution_path, "temp_input_image.jpg")
    temp_output_path = os.path.join(execution_path, "temp_output_image.jpg")

    with open(temp_input_path, "wb") as temp_file:
        temp_file.write(image_bytes)

    # Perform detection
    detections = detector.detectObjectsFromImage(
        input_image=temp_input_path,
        output_image_path=temp_output_path,
        minimum_percentage_probability=30
    )

    # Process results
    results = []
    for eachObject in detections:
        results.append({
            "name": eachObject["name"],
            "percentage_probability": eachObject["percentage_probability"],
            "box_points": eachObject["box_points"]
        })

    # Clean up temporary files
    # os.remove(temp_input_path)
    # os.remove(temp_output_path)

    return results
