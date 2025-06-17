# ==============================================================================
# AI MODEL MANAGEMENT
# ==============================================================================
import numpy as np

NUM_KEYPOINTS = 14  # Number of keypoints per person

try:
    from rtmlib import RTMO  # type: ignore
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False
    print("WARNING: rtmlib not found. AI Pose Estimation will be disabled.")

class RTMOManager:
    """Manages the RTMO pose estimation model loading and inference."""
    def __init__(self, model_path: str, input_size: tuple[int, int], backend: str = 'onnxruntime', device: str = 'cuda'):
        """
        Initializes the RTMO model.
        Args:
            model_path: Path to the ONNX model file.
            input_size: Expected input image size (width, height) for the model.
            backend: Inference backend (e.g., 'onnxruntime').
            device: Device for inference ('cuda' or 'cpu').
        """
        self.model = None
        self.input_size = input_size
        if RTMLIB_AVAILABLE:
            try:
                self.model = RTMO(model_path, model_input_size=input_size, backend=backend, device=device)
                print(f"RTMO model loaded successfully from {model_path} on {device}.")
            except Exception as e:
                print(f"ERROR: Failed to load RTMO model from {model_path}: {e}")
                self.model = None # Ensure model is None on failure
        else:
            print("INFO: RTMOManager initialized, but rtmlib is not available (ImportError). AI features will be disabled.")

    def is_ready(self) -> bool:
        """Checks if the model was loaded successfully and is ready for inference."""
        return self.model is not None

    def predict_poses(self, frame_bgr: np.ndarray) -> list[dict] | None:
        """
        Performs pose estimation on a single BGR frame using the RTMO model.
        Args:
            frame_bgr: The input image frame in BGR format.
        Returns:
            A list of dictionaries, where each dictionary represents a detected person
            and contains 'keypoints' (NxNUM_KEYPOINTSx2) and 'scores' (NxNUM_KEYPOINTS).
            Returns None on error or if the model is not ready. Returns empty list if no people detected.
        """
        if not self.is_ready() or self.model is None:
            print("RTMO model not ready for prediction.")
            return None
        try:
            # `model()` call expects BGR numpy array
            kps_batch, scores_batch = self.model(frame_bgr) # kps_batch: (N, K, 2), scores_batch: (N, K)

            processed_results = []
            if kps_batch is None or scores_batch is None:
                # This can happen if the model internally decides no confident poses were found.
                print("AI Pose Estimation: Model returned None for keypoints or scores (no poses detected or low confidence).")
                return [] # Return empty list, not None, to indicate successful run with no detections

            # Basic validation of return types and shapes
            if not isinstance(kps_batch, np.ndarray) or not isinstance(scores_batch, np.ndarray):
                print(f"AI Pose Estimation: Unexpected type from model. Keypoints: {type(kps_batch)}, Scores: {type(scores_batch)}")
                return []

            num_people = kps_batch.shape[0]
            if num_people == 0:
                return [] # No people detected

            # Validate dimensions based on expected output
            if not (kps_batch.ndim == 3 and kps_batch.shape[1] == NUM_KEYPOINTS and kps_batch.shape[2] == 2):
                print(f"AI Pose: Unexpected kps_batch shape. Expected (N, {NUM_KEYPOINTS}, 2), got {kps_batch.shape}")
                return []
            if not (scores_batch.ndim == 2 and scores_batch.shape[1] == NUM_KEYPOINTS):
                print(f"AI Pose: Unexpected scores_batch shape. Expected (N, {NUM_KEYPOINTS}), got {scores_batch.shape}")
                return []
            if kps_batch.shape[0] != scores_batch.shape[0]:
                print("AI Pose: Mismatch in number of people detected between keypoints and scores.")
                return []

            for i in range(num_people):
                processed_results.append({
                    'keypoints': kps_batch[i],    # Keypoints for person i (NUM_KEYPOINTS, 2)
                    'scores': scores_batch[i],    # Scores for person i (NUM_KEYPOINTS,)
                    'bbox_ai': None               # Placeholder for bbox; rtmlib RTMO might not directly return it here
                                                  # but some models/wrappers might.
                })
            return processed_results

        except Exception as e:
            print(f"Error during RTMO pose estimation call or processing: {e}")
            import traceback
            traceback.print_exc()
            return None # Indicate failure