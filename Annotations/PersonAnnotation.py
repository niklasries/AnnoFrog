# ==============================================================================
# CORE DATA STRUCTURES
# ==============================================================================
NUM_KEYPOINTS = 14  # Number of keypoints per person

# Keypoint visibility states
VISIBILITY_NOT_SET = 0        # Keypoint not yet annotated
VISIBILITY_OCCLUDED = 1       # Keypoint annotated as occluded
VISIBILITY_VISIBLE = 2        # Keypoint annotated as visible
VISIBILITY_SUGGESTED = 3      # Keypoint is an interpolated suggestion
VISIBILITY_AI_SUGGESTED = 4   # Keypoint is an AI-generated suggestion


class PersonAnnotation:
    """Stores annotation data (ID, bbox, keypoints) for a single person in a single frame."""
    def __init__(self, person_id: int, bbox: list[float] | None = None,
                 keypoints: list[list[float | int]] | None = None):
        """
        Initializes a person's annotation.
        Args:
            person_id: Unique identifier for the person track.
            bbox: Bounding box [x_min, y_min, x_max, y_max] in normalized image coordinates.
            keypoints: List of keypoints, each [x, y, visibility_flag].
        """
        self.id: int = person_id
        self.bbox: list[float] = bbox if bbox else [0.0, 0.0, 0.0, 0.0]  # Normalized [x_min, y_min, x_max, y_max]
        # Initialize keypoints with VISIBILITY_NOT_SET
        self.keypoints: list[list[float|int]] = keypoints if keypoints else \
                                               [[0.0, 0.0, VISIBILITY_NOT_SET] for _ in range(NUM_KEYPOINTS)]

    def to_dict(self) -> dict:
        """Converts the annotation to a dictionary for JSON serialization."""
        return {"id": self.id, "bbox": self.bbox, "keypoints": self.keypoints}

    @classmethod
    def from_dict(cls, data: dict) -> 'PersonAnnotation':
        """Creates a PersonAnnotation instance from a dictionary (e.g., from JSON)."""
        return cls(data["id"], data["bbox"], data["keypoints"])

    def is_keypoint_set(self, kp_idx: int) -> bool:
        """Checks if a specific keypoint has been set (i.e., not VISIBILITY_NOT_SET)."""
        return self.keypoints[kp_idx][2] != VISIBILITY_NOT_SET

    def get_next_unset_keypoint_idx(self, start_idx: int = 0) -> int | None:
        """
        Finds the index of the next keypoint that is VISIBILITY_NOT_SET,
        starting from start_idx and wrapping around.
        """
        for i in range(NUM_KEYPOINTS):
            current_kp_idx = (start_idx + i) % NUM_KEYPOINTS
            if self.keypoints[current_kp_idx][2] == VISIBILITY_NOT_SET:
                return current_kp_idx
        return None # All keypoints are set

    def all_keypoints_set(self) -> bool:
        """Checks if all keypoints have been explicitly set by the user (Visible or Occluded)."""
        return all(self.keypoints[i][2] in [VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE] for i in range(NUM_KEYPOINTS))

    def has_suggestions(self) -> bool:
        """Checks if any keypoint in this annotation is currently an interpolated suggestion."""
        return any(kp[2] == VISIBILITY_SUGGESTED for kp in self.keypoints)

    def has_ai_suggestions(self) -> bool:
        """Checks if any keypoint in this annotation is an AI-generated suggestion."""
        return any(kp[2] == VISIBILITY_AI_SUGGESTED for kp in self.keypoints)

    def is_suggestion_any_type(self) -> bool:
        """Checks if any keypoint is either an interpolated or AI-generated suggestion."""
        return any(kp[2] in [VISIBILITY_SUGGESTED, VISIBILITY_AI_SUGGESTED] for kp in self.keypoints)