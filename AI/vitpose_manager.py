# vitpose_manager.py
import torch
from PIL import Image
import numpy as np
import cv2 
from transformers import AutoProcessor, VitPoseForPoseEstimation

# Default model, can be overridden in constructor
DEFAULT_VITPOSE_MODEL_NAME = 'usyd-community/vitpose-base-coco-aic-mpii'
# This model outputs 17 COCO keypoints.
VITPOSE_COCO_BODY_INDICES_TO_KEEP = [
    5,  # L_Shoulder
    6,  # R_Shoulder
    7,  # L_Elbow
    8,  # R_Elbow
    9,  # L_Wrist
    10, # R_Wrist
    11, # L_Hip
    12, # R_Hip
    13, # L_Knee
    14, # R_Knee
    15, # L_Ankle
    16, # R_Ankle
]

NUM_VITPOSE_BODY_KEYPOINTS_TO_USE = len(VITPOSE_COCO_BODY_INDICES_TO_KEEP)

class ViTPoseManager:
    def __init__(self, model_name: str = DEFAULT_VITPOSE_MODEL_NAME, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if not torch.cuda.is_available() and self.device == 'cuda':
            print("WARNING: CUDA not available, ViTPose falling back to CPU.")
            self.device = 'cpu'
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = VitPoseForPoseEstimation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"ViTPose model '{self.model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"ERROR: Failed to load ViTPose model '{self.model_name}': {e}")
            self.processor = None
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def predict_poses_from_bboxes(self, frame_numpy_bgr: np.ndarray,
                                  user_bboxes_pixel_xywh: list[list[float]]) -> list[dict] | None:
        if not self.is_ready():
            print("ViTPose model not ready for prediction.")
            return None
        if not user_bboxes_pixel_xywh:
            return []

        try:
            frame_rgb = cv2.cvtColor(frame_numpy_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inputs = self.processor(
                images=pil_image,
                boxes=[user_bboxes_pixel_xywh],
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            pose_results_batch = self.processor.post_process_pose_estimation(
                outputs, boxes=[user_bboxes_pixel_xywh]
            )

            if not pose_results_batch: return []
                
            person_pose_results = pose_results_batch[0]
            processed_results = []
            for person_result in person_pose_results:
                raw_keypoints_pixel = person_result['keypoints'].cpu().numpy() # (17, 2) for COCO
                raw_scores = person_result['scores'].cpu().numpy()           # (17,)

                if raw_keypoints_pixel.shape[0] < 17: # Ensure we got enough for COCO
                    print(f"Warning: ViTPose returned {raw_keypoints_pixel.shape[0]} keypoints, expected 17. Skipping.")
                    continue

                selected_kps = raw_keypoints_pixel[VITPOSE_COCO_BODY_INDICES_TO_KEEP] # Shape (12, 2)
                selected_scores = raw_scores[VITPOSE_COCO_BODY_INDICES_TO_KEEP]     # Shape (12,)

                processed_results.append({
                    'keypoints': selected_kps, # 12 body keypoints in pixel coords
                    'scores': selected_scores,   # Scores for these 12
                    'source_model': 'vitpose'       # Add a flag
                })
            return processed_results
        except Exception as e:
            print(f"Error during ViTPose prediction: {e}")
            import traceback
            traceback.print_exc()
            return None