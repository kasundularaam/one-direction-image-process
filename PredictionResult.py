class PredictionResult:
    def __init__(self, cls, confident, x_min, y_min, x_max, y_max):
        self.cls = cls
        self.confident = confident
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __repr__(self):
        return (f"PredictionResult(class='{self.cls}', confidence={self.confident:.2f}, "
                f"bbox=({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}))")
