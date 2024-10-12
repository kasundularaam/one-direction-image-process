from ultralytics import YOLO

model = YOLO("best.pt")


def check_forward_centering(result, image_width, tolerance=0.1):
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
            centering_info.append({
                'is_centered': is_centered,
                'displacement': displacement,
                'confidence': float(box.conf[0])
            })

    return centering_info


def process(image):
    results = model(image)
    image_width = results[0].orig_shape[1]  # Get the original image width

    for result in results:
        centering_info = check_forward_centering(result, image_width)

        if centering_info:
            for info in centering_info:

                displacement = round(info['displacement'])*(326/10)

                if (info['confidence'] > 0.9):
                    pass

                print(f"  Centered: {info['is_centered']}")
                print(f"  Displacement: {info['displacement']:.2f} pixels")
                print(f"  Confidence: {info['confidence']:.2f}")
                print("---")
        else:
            print("No 'forward' objects detected.")
