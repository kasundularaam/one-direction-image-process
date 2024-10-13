from ultralytics import YOLO

model = YOLO("best.pt")


class PredictionResult:
    def __init__(self, direction, is_centered,  displacement, confidence):
        self.direction = direction
        self.is_centered = is_centered
        self.displacement = displacement
        self.confidence = confidence

    def where_to_go(self):
        if (self.is_centered):
            return "forward"
        if (self.displacement > 0):
            return "right"
        return "left"

    def __str__(self):
        return f"Direction: {self.direction}, Is Centered: {self.is_centered},  Displacement: {self.displacement}, Confidence: {self.confidence}"


def check_forward_centering(result, image_width, tolerance=0.1) -> PredictionResult:
    """
    Check if detected 'forward' classes are centered and calculate displacement.

    :param result: A single result from YOLO model
    :param image_width: Width of the original image
    :param tolerance: Tolerance for considering an object centered (0.1 = 10% of image width)
    :return: List of dictionaries containing centering information for each 'forward' detection
    """
    centering_info = []

    # Get the center x-coordinate of the image
    image_center_x = image_width / 2

    # Iterate through all detected objects
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        # Check if the detected class is 'forward'
        if class_name == 'forward':
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate center of the bounding box
            box_center_x = (x1 + x2) / 2

            # Calculate displacement from image center
            displacement = box_center_x - image_center_x

            # Check if the box is centered within the tolerance
            is_centered = abs(displacement) <= (image_width * tolerance)

            # Add information to the list
            centering_info.append(
                PredictionResult(
                    direction="forward",
                    confidence=float(box.conf[0]),
                    is_centered=is_centered,
                    displacement=displacement,
                )
            )

    return max(centering_info, key=lambda x: x.confidence)


def process(image):
    results = model(image)
    image_width = results[0].orig_shape[1]  # Get the original image width

    result = results[0]

    most_confident = check_forward_centering(result, image_width)
    result.save(filename="res.jpg")

    return most_confident.where_to_go()
