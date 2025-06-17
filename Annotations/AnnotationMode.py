from enum import Enum, auto

class AnnotationMode(Enum):
    """Defines the current interaction mode of the annotation canvas."""
    IDLE = auto()               # No active annotation task
    CREATING_BBOX_P1 = auto()   # Defining the first corner of a new bounding box
    CREATING_BBOX_P2 = auto()   # Defining the second corner of a new bounding box
    PLACING_KEYPOINTS = auto()  # Placing or editing keypoints for an active person