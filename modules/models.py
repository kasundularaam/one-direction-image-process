from dataclasses import dataclass
from typing import Optional


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
