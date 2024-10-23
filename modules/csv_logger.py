import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, filename="data.csv"):
        self.filename = filename
        self.initialize_csv()

    def initialize_csv(self):
        """Create CSV file with ML-friendly headers"""
        if not Path(self.filename).exists():
            with open(self.filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    # Features
                    'arrow_count',
                    # Arrow 1 features
                    'has_left_1',
                    'has_right_1',
                    'has_forward_1',
                    'confidence_1',
                    'x_min_1',
                    'y_min_1',
                    'x_max_1',
                    'y_max_1',
                    # Arrow 2 features
                    'has_left_2',
                    'has_right_2',
                    'has_forward_2',
                    'confidence_2',
                    'x_min_2',
                    'y_min_2',
                    'x_max_2',
                    'y_max_2',
                    # Arrow 3 features
                    'has_left_3',
                    'has_right_3',
                    'has_forward_3',
                    'confidence_3',
                    'x_min_3',
                    'y_min_3',
                    'x_max_3',
                    'y_max_3',
                    # Target
                    'direction'
                ])

    def log_result(self, result):
        """Log a single result to CSV in ML-friendly format"""
        def get_arrow_features(label, conf, x_min, y_min, x_max, y_max):
            """Helper function to compute features for one arrow"""
            # One-hot encoding for arrow direction
            is_left = 1 if label == 'left' else 0
            is_right = 1 if label == 'right' else 0
            is_forward = 1 if label == 'forward' else 0

            return [is_left, is_right, is_forward,
                    conf if conf is not None else 0,
                    x_min if x_min is not None else 0,
                    y_min if y_min is not None else 0,
                    x_max if x_max is not None else 0,
                    y_max if y_max is not None else 0]

        # Get features for each arrow
        arrow1_features = get_arrow_features(
            result.arrow_label1, result.confidence1,
            result.x_min1, result.y_min1, result.x_max1, result.y_max1
        )

        arrow2_features = get_arrow_features(
            result.arrow_label2, result.confidence2,
            result.x_min2, result.y_min2, result.x_max2, result.y_max2
        )

        arrow3_features = get_arrow_features(
            result.arrow_label3, result.confidence3,
            result.x_min3, result.y_min3, result.x_max3, result.y_max3
        )

        # Combine all features
        row = [
            result.arrow_count,
            *arrow1_features,  # Unpack arrow 1 features
            *arrow2_features,  # Unpack arrow 2 features
            *arrow3_features,  # Unpack arrow 3 features
            result.direction   # Target variable
        ]

        # Write to CSV
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
