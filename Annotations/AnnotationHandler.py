import numpy as np
import json


# --- OpenGL Imports ---
from OpenGL.GL import *

# --- PyQt6 Imports ---
from PyQt6.QtCore import pyqtSignal, QObject



NUM_KEYPOINTS = 14  # Number of keypoints per person

# Keypoint visibility states
VISIBILITY_NOT_SET = 0        # Keypoint not yet annotated
VISIBILITY_OCCLUDED = 1       # Keypoint annotated as occluded
VISIBILITY_VISIBLE = 2        # Keypoint annotated as visible
VISIBILITY_SUGGESTED = 3      # Keypoint is an interpolated suggestion
VISIBILITY_AI_SUGGESTED = 4   # Keypoint is an AI-generated suggestion



from Annotations.PersonAnnotation import PersonAnnotation

class AnnotationHandler(QObject):
    """
    Manages loading, saving, and accessing all annotations for a video,
    including 'done' states for person tracks and suggested annotations.
    """
    real_annotations_changed = pyqtSignal()  # Emitted when real annotations are modified (saved, loaded, changed)
    done_state_changed = pyqtSignal(int, bool) # Emitted when a person's 'done' state changes: (person_id, is_done_status)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        # Stores real (user-confirmed) annotations: {frame_idx: [PersonAnnotation, ...]}
        self.all_annotations_by_frame: dict[int, list[PersonAnnotation]] = {}
        # Stores suggested (interpolated or AI) annotations: {frame_idx: [PersonAnnotation, ...]}
        self.suggested_annotations_by_frame: dict[int, list[PersonAnnotation]] = {}
        self.current_json_path: str | None = None  # Path to the .json annotation file
        self.done_person_ids: set[int] = set() # Stores IDs of persons marked as 'done' globally

    def load_annotations_for_video(self, video_filepath: str):
        """Loads annotations and 'done' states from a JSON file associated with the video."""
        self.current_json_path = video_filepath.rsplit('.', 1)[0] + "_annotations.json"
        self.all_annotations_by_frame.clear()
        self.clear_all_suggestions(preserve_ai_suggestions=False) # Clear all types of suggestions
        self.done_person_ids.clear()
        try:
            with open(self.current_json_path, 'r') as f:
                data = json.load(f)
                raw_annotations = data.get("annotations", {})
                self.all_annotations_by_frame = {
                    int(frame_idx_str): [PersonAnnotation.from_dict(p_data) for p_data in persons_list]
                    for frame_idx_str, persons_list in raw_annotations.items()
                }
                self.done_person_ids = set(data.get("done_ids", []))
            print(f"Loaded annotations and done IDs from {self.current_json_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Annotation file issue ({e}). Starting fresh for {self.current_json_path}")
            self.all_annotations_by_frame = {} # Initialize empty if file not found or corrupt
            self.done_person_ids = set()
        self.real_annotations_changed.emit()

    def save_annotations(self):
        """Saves all real annotations and 'done' states to the current JSON file."""
        if not self.current_json_path:
            print("Warning: Annotation save attempted but no JSON path is set.")
            return

        annotations_to_save = {
            str(frame_idx): [p.to_dict() for p in persons_list if p] # Ensure no None persons
            for frame_idx, persons_list in self.all_annotations_by_frame.items() if persons_list
        }
        data_to_save_globally = {
            "annotations": annotations_to_save,
            "done_ids": sorted(list(self.done_person_ids)) # Save sorted list of done IDs
        }
        try:
            with open(self.current_json_path, 'w') as f:
                json.dump(data_to_save_globally, f, indent=2)
            print(f"Saved annotations and done IDs to {self.current_json_path}")
        except IOError as e:
            print(f"ERROR: Could not save annotations to {self.current_json_path}: {e}")
        self.real_annotations_changed.emit()

    def get_annotations_for_frame(self, frame_idx: int) -> list[PersonAnnotation]:
        """Retrieves all real annotations for a given frame index."""
        return self.all_annotations_by_frame.get(frame_idx, [])

    def get_suggested_annotations_for_frame(self, frame_idx: int) -> list[PersonAnnotation]:
        """Retrieves all suggested (interpolated or AI) annotations for a given frame index."""
        return self.suggested_annotations_by_frame.get(frame_idx, [])

    def clear_all_suggestions(self, preserve_ai_suggestions: bool = False):
        """
        Clears stored suggested annotations.
        Args:
            preserve_ai_suggestions: If True, AI-generated suggestions are kept, others removed.
                                     If False, all suggestions are removed.
        """
        if not self.suggested_annotations_by_frame:
            return

        if preserve_ai_suggestions:
            frames_to_remove_if_empty_after_filter = []
            for frame_idx, suggestions_on_frame in list(self.suggested_annotations_by_frame.items()):
                ai_suggestions_kept = [sugg for sugg in suggestions_on_frame if sugg.has_ai_suggestions()]
                if ai_suggestions_kept:
                    self.suggested_annotations_by_frame[frame_idx] = ai_suggestions_kept
                else:
                    # Mark frame for deletion if no AI suggestions were kept
                    frames_to_remove_if_empty_after_filter.append(frame_idx)

            for frame_idx_to_delete in frames_to_remove_if_empty_after_filter:
                if frame_idx_to_delete in self.suggested_annotations_by_frame:
                    del self.suggested_annotations_by_frame[frame_idx_to_delete]
            if frames_to_remove_if_empty_after_filter:
                 print("Cleared interpolated suggestions, preserved AI suggestions.")
        else:
            if self.suggested_annotations_by_frame:
                self.suggested_annotations_by_frame.clear()
                print("Cleared all types of suggestions (interpolated and AI).")

    def add_person_to_frame(self, frame_idx: int, person_annotation: PersonAnnotation):
        """
        Adds or updates a person's real annotation for a frame.
        If person_annotation.id is -1 or doesn't exist on frame, a new ID is assigned.
        If person_annotation.id exists, it's updated.
        Also removes any existing suggestion for the same person on this frame.
        """
        if frame_idx not in self.all_annotations_by_frame:
            self.all_annotations_by_frame[frame_idx] = []

        current_persons_on_frame = self.all_annotations_by_frame[frame_idx]
        is_update_of_existing_id = False

        if person_annotation.id >= 0: # Check if a specific ID is provided
            existing_person_obj = next((p for p in current_persons_on_frame if p.id == person_annotation.id), None)
            if existing_person_obj:
                current_persons_on_frame.remove(existing_person_obj) # Remove to re-add (update)
                is_update_of_existing_id = True

        if not is_update_of_existing_id: # Assign a new ID if it's a new person or ID was -1
            new_id = 0
            if current_persons_on_frame: # Find max existing ID on this frame
                max_id_found = -1
                for p_obj in current_persons_on_frame:
                    if isinstance(p_obj.id, int) and p_obj.id > max_id_found:
                        max_id_found = p_obj.id
                new_id = max_id_found + 1
            person_annotation.id = new_id

        current_persons_on_frame.append(person_annotation)
        # self.all_annotations_by_frame[frame_idx] is already referencing current_persons_on_frame

        # If adding/updating a real annotation, remove any existing suggestion for the same person_id on this frame
        if frame_idx in self.suggested_annotations_by_frame:
            self.suggested_annotations_by_frame[frame_idx] = [
                s_ann for s_ann in self.suggested_annotations_by_frame[frame_idx]
                if s_ann.id != person_annotation.id
            ]
            if not self.suggested_annotations_by_frame[frame_idx]: # If list becomes empty
                del self.suggested_annotations_by_frame[frame_idx]

    def get_person_by_id_in_frame(self, frame_idx: int, person_id: int) -> PersonAnnotation | None:
        """Retrieves a specific real person annotation by ID from a given frame."""
        return next((p for p in self.get_annotations_for_frame(frame_idx) if p.id == person_id), None)

    def generate_interpolation_suggestions(self, start_frame_idx: int, person_id: int,
                                           max_gap_to_search_for_end_kf: int, interp_type: str,
                                           video_num_frames: int) -> tuple[int, int] | None:
        """
        Generates interpolated keypoint suggestions between two real keyframes of a person.
        Args:
            start_frame_idx: The frame index of the starting keyframe.
            person_id: The ID of the person to interpolate.
            max_gap_to_search_for_end_kf: Max frames to search forward for the next keyframe.
            interp_type: Type of interpolation (e.g., "Linear").
            video_num_frames: Total number of frames in the video.
        Returns:
            A tuple (first_interpolated_frame_idx, last_interpolated_frame_idx) if successful, else None.
        """
        # 1. Find the starting person keyframe
        person_start = self.get_person_by_id_in_frame(start_frame_idx, person_id)
        if not person_start or not any(kp[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED] for kp in person_start.keypoints):
            # Start keyframe must exist and have at least one user-set keypoint
            return None

        # 2. Find the ending person keyframe within the max_gap
        person_end = None
        end_frame_idx = -1
        for i in range(1, max_gap_to_search_for_end_kf + 1):
            current_check_frame_idx = start_frame_idx + i
            if current_check_frame_idx >= video_num_frames: break # Don't exceed video length
            p_on_frame = self.get_person_by_id_in_frame(current_check_frame_idx, person_id)
            if p_on_frame and any(kp[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED] for kp in p_on_frame.keypoints):
                person_end = p_on_frame
                end_frame_idx = current_check_frame_idx
                break
        if not person_end: return None # No suitable end keyframe found

        # 3. Generate interpolations for frames in between
        num_interpolated_frames = 0
        for frame_k in range(start_frame_idx + 1, end_frame_idx):
            # Skip if a real annotation for this person already exists on this intermediate frame
            if self.get_person_by_id_in_frame(frame_k, person_id): continue

            # Initialize suggestion list for this frame if it doesn't exist
            if frame_k not in self.suggested_annotations_by_frame:
                self.suggested_annotations_by_frame[frame_k] = []

            # Skip if a suggestion for this person_id already exists on this frame (e.g., from AI)
            if any(s_p.id == person_id for s_p in self.suggested_annotations_by_frame[frame_k]):
                continue

            suggested_person = PersonAnnotation(person_id=person_id, bbox=list(person_start.bbox)) # Use start bbox for now
            t = (frame_k - start_frame_idx) / (end_frame_idx - start_frame_idx) # Interpolation factor
            
            # --- Bounding Box Interpolation ---
            start_bbox_coords = person_start.bbox
            end_bbox_coords = person_end.bbox

            # Interpolate bbox if both start and end bboxes are defined (not [0,0,0,0])
            start_bbox_defined = not (abs(start_bbox_coords[0]) < 1e-6 and abs(start_bbox_coords[1]) < 1e-6 and \
                                      abs(start_bbox_coords[2]) < 1e-6 and abs(start_bbox_coords[3]) < 1e-6)
            end_bbox_defined = not (abs(end_bbox_coords[0]) < 1e-6 and abs(end_bbox_coords[1]) < 1e-6 and \
                                    abs(end_bbox_coords[2]) < 1e-6 and abs(end_bbox_coords[3]) < 1e-6)

            if start_bbox_defined and end_bbox_defined:
                interp_bbox_x_min = (1 - t) * start_bbox_coords[0] + t * end_bbox_coords[0]
                interp_bbox_y_min = (1 - t) * start_bbox_coords[1] + t * end_bbox_coords[1]
                interp_bbox_x_max = (1 - t) * start_bbox_coords[2] + t * end_bbox_coords[2]
                interp_bbox_y_max = (1 - t) * start_bbox_coords[3] + t * end_bbox_coords[3]
                suggested_person.bbox = [interp_bbox_x_min, interp_bbox_y_min, interp_bbox_x_max, interp_bbox_y_max]
            elif start_bbox_defined: # If only start is defined, carry it forward (or choose other logic)
                suggested_person.bbox = list(start_bbox_coords)
            else: # Otherwise, default to an empty bbox for the suggestion
                suggested_person.bbox = [0.0, 0.0, 0.0, 0.0]
            # --- End Bounding Box Interpolation ---

            # TODO: Implement different interpolation types (currently linear)
            # if interp_type == "Cubic": ...
            for kp_idx in range(NUM_KEYPOINTS):
                kp_s, kp_e = person_start.keypoints[kp_idx], person_end.keypoints[kp_idx]
                # Interpolate only if both start and end keypoints are user-set (visible or occluded)
                if kp_s[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED] and \
                   kp_e[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED]:
                    x_interp = (1 - t) * kp_s[0] + t * kp_e[0]
                    y_interp = (1 - t) * kp_s[1] + t * kp_e[1]
                    suggested_person.keypoints[kp_idx] = [x_interp, y_interp, VISIBILITY_SUGGESTED]
                else:
                    # If one of the keypoints isn't set, don't interpolate this specific keypoint
                    suggested_person.keypoints[kp_idx] = [0.0, 0.0, VISIBILITY_NOT_SET]

            self.suggested_annotations_by_frame[frame_k].append(suggested_person)
            num_interpolated_frames +=1

        if num_interpolated_frames > 0:
            return (start_frame_idx + 1, end_frame_idx - 1) # Range of frames where suggestions were added
        return None

    def accept_suggestion_for_frame(self, frame_idx: int, person_id: int, is_auto_interpolate_on: bool) -> PersonAnnotation | None:
        """
        Converts a suggested annotation (interpolated or AI) into a real one for a specific person on a frame.
        Handles cleanup of the suggestion list and potentially other suggestions for the same person
        if auto-interpolation is off.
        Args:
            frame_idx: The frame index where the suggestion exists.
            person_id: The ID of the person whose suggestion is to be accepted.
            is_auto_interpolate_on: Flag indicating if auto-interpolation is active.
        Returns:
            The newly created real PersonAnnotation if successful, else None.
        """
        original_suggestions_on_frame = self.suggested_annotations_by_frame.get(frame_idx, [])
        suggestion_to_accept: PersonAnnotation | None = None
        remaining_suggestions_on_frame: list[PersonAnnotation] = []
        was_suggestion_found = False

        # Find and separate the suggestion to accept
        for sugg in original_suggestions_on_frame:
            if sugg.id == person_id:
                suggestion_to_accept = sugg
                was_suggestion_found = True
            else:
                remaining_suggestions_on_frame.append(sugg)

        if not was_suggestion_found or suggestion_to_accept is None:
            print(f"Warning: Tried to accept suggestion for P{person_id} on frame {frame_idx}, but it was not found.")
            return None

        # Update the suggestions list for the current frame immediately
        if not remaining_suggestions_on_frame: # No other suggestions left on this frame
            if frame_idx in self.suggested_annotations_by_frame: del self.suggested_annotations_by_frame[frame_idx]
        else:
            self.suggested_annotations_by_frame[frame_idx] = remaining_suggestions_on_frame

        # Create the real annotation from the (now removed from suggestions) suggestion
        real_person_ann = PersonAnnotation.from_dict(suggestion_to_accept.to_dict())
        for kp_data in real_person_ann.keypoints:
            if kp_data[2] in [VISIBILITY_SUGGESTED, VISIBILITY_AI_SUGGESTED]:
                kp_data[2] = VISIBILITY_VISIBLE # Default accepted suggestions to visible

        # Add/Update the real annotation.
        # This will replace an existing real annotation with the same ID on this frame or add new.
        current_real_anns_on_frame = self.all_annotations_by_frame.get(frame_idx, [])
        existing_real_person = next((p for p in current_real_anns_on_frame if p.id == real_person_ann.id), None)
        if existing_real_person:
            current_real_anns_on_frame.remove(existing_real_person)
        current_real_anns_on_frame.append(real_person_ann)
        self.all_annotations_by_frame[frame_idx] = current_real_anns_on_frame

        # If auto-interpolate is OFF, accepting a suggestion clears ALL other suggestions
        # (interpolated or AI) for this person_id across ALL frames.
        if not is_auto_interpolate_on:
            frames_to_remove_sugg_list_from = []
            for f_idx_iter, sugg_list_iter in list(self.suggested_annotations_by_frame.items()):
                # Filter out suggestions for the accepted person_id
                filtered_sugg_list = [s for s in sugg_list_iter if s.id != person_id]
                if len(filtered_sugg_list) < len(sugg_list_iter): # If something was removed
                    if not filtered_sugg_list: # If the list becomes empty
                        frames_to_remove_sugg_list_from.append(f_idx_iter)
                    else:
                        self.suggested_annotations_by_frame[f_idx_iter] = filtered_sugg_list
            for f_idx_to_delete in frames_to_remove_sugg_list_from:
                if f_idx_to_delete in self.suggested_annotations_by_frame:
                    del self.suggested_annotations_by_frame[f_idx_to_delete]
            if frames_to_remove_sugg_list_from or was_suggestion_found : # Log if any changes were made
                print(f"Cleared all suggestions for P{person_id} as auto-interpolate is off and a suggestion was accepted.")

        self.save_annotations() # Persist changes
        return real_person_ann

    def discard_suggestion_for_frame(self, frame_idx: int, person_id: int):
        """Removes a suggested annotation for a specific person on a specific frame."""
        if frame_idx in self.suggested_annotations_by_frame:
            original_len = len(self.suggested_annotations_by_frame[frame_idx])
            self.suggested_annotations_by_frame[frame_idx] = [
                p for p in self.suggested_annotations_by_frame[frame_idx] if p.id != person_id
            ]
            if not self.suggested_annotations_by_frame[frame_idx]: # If list becomes empty
                del self.suggested_annotations_by_frame[frame_idx]

            if len(self.suggested_annotations_by_frame.get(frame_idx, [])) < original_len:
                 print(f"Discarded suggestion for P{person_id} on frame {frame_idx+1}")
                 # No save here, as discarding is usually a transient UI action before other edits or acceptances.
                 # Save will happen when a real annotation is made or on explicit save.

    def add_ai_pose_suggestions_to_frame(self, frame_idx: int,
                                         raw_ai_poses: list[dict],
                                         kp_confidence_threshold: float,
                                         target_mode: str,
                                         user_bboxes_for_empty_mode: list[PersonAnnotation] | None = None,
                                         frame_width: int = 0, frame_height: int = 0) -> int:
        """
        Processes raw AI pose data and adds them as AI suggestions to the frame.
        Args:
            frame_idx: Index of the frame to add suggestions to.
            raw_ai_poses: List of dictionaries, each from RTMOManager.predict_poses.
            kp_confidence_threshold: Minimum score for a keypoint to be considered valid.
            target_mode: "All Detected People" or "Only for Empty User BBoxes".
            user_bboxes_for_empty_mode: List of PersonAnnotation (real) on the frame, used if target_mode is "Only for Empty User BBoxes".
            frame_width: Width of the video frame in pixels.
            frame_height: Height of the video frame in pixels.
        Returns:
            Number of AI pose suggestions successfully added to the frame.
        """
        if not raw_ai_poses: return 0
        if frame_idx not in self.suggested_annotations_by_frame:
            self.suggested_annotations_by_frame[frame_idx] = []

        num_ai_suggestions_added = 0

        # Determine a starting ID for new AI-detected people to avoid conflicts
        max_existing_id = -1
        for frame_anns_list in self.all_annotations_by_frame.values(): # Check real annotations
            for p_ann in frame_anns_list: max_existing_id = max(max_existing_id, p_ann.id)
        for frame_suggs_list in self.suggested_annotations_by_frame.values(): # Check existing suggestions
            for p_sugg in frame_suggs_list: max_existing_id = max(max_existing_id, p_sugg.id)
        next_ai_id_candidate = max(max_existing_id + 1, 10000) # Start AI-generated IDs high to distinguish

        for ai_pose_data in raw_ai_poses:
            ai_keypoints_pixel = ai_pose_data['keypoints']  # Shape (NUM_KEYPOINTS, 2)
            ai_keypoint_scores = ai_pose_data['scores']    # Shape (NUM_KEYPOINTS,)
            ai_detected_bbox_pixel = ai_pose_data.get('bbox_ai') # Optional, [x1,y1,x2,y2] in pixels

            # Filter by number of confident keypoints
            num_confident_kps = np.sum(ai_keypoint_scores >= kp_confidence_threshold)
            if num_confident_kps < 3: # Arbitrary threshold, tune as needed
                continue

            # Normalize keypoints and set visibility
            normalized_keypoints_data = [[0.0,0.0,VISIBILITY_NOT_SET] for _ in range(NUM_KEYPOINTS)]
            num_valid_normalized_kps = 0
            if frame_width > 0 and frame_height > 0:
                for i in range(NUM_KEYPOINTS):
                    if ai_keypoint_scores[i] >= kp_confidence_threshold:
                        norm_x = max(0.0, min(1.0, ai_keypoints_pixel[i,0] / frame_width))
                        norm_y = max(0.0, min(1.0, ai_keypoints_pixel[i,1] / frame_height))
                        normalized_keypoints_data[i] = [norm_x, norm_y, VISIBILITY_AI_SUGGESTED]
                        num_valid_normalized_kps +=1
            else:
                print("Warning: Frame dimensions missing for AI suggestion normalization.")
                continue # Cannot normalize without frame dimensions
            if num_valid_normalized_kps < 3 : continue # If after normalization, not enough valid kps

            # Determine bounding box (from AI detection or from keypoints)
            bbox_norm_from_ai = [0.0,0.0,0.0,0.0]
            if ai_detected_bbox_pixel is not None and frame_width > 0 and frame_height > 0:
                # Normalize AI detected bounding box
                bbox_norm_from_ai = [ai_detected_bbox_pixel[0]/frame_width, ai_detected_bbox_pixel[1]/frame_height,
                                     ai_detected_bbox_pixel[2]/frame_width, ai_detected_bbox_pixel[3]/frame_height]
            else: # If no bbox from AI, derive from confident keypoints
                confident_kps_norm_coords = np.array([kp[0:2] for kp in normalized_keypoints_data if kp[2]==VISIBILITY_AI_SUGGESTED])
                if confident_kps_norm_coords.shape[0] > 0:
                    min_x,min_y = np.min(confident_kps_norm_coords,axis=0)
                    max_x,max_y = np.max(confident_kps_norm_coords,axis=0)
                    bbox_norm_from_ai = [min_x,min_y,max_x,max_y]
                # If no confident keypoints, bbox_norm_from_ai remains [0,0,0,0]

            should_add_this_ai_pose = False
            id_for_this_pose = next_ai_id_candidate # Tentative ID for "All Detected" mode
            bbox_for_this_pose = bbox_norm_from_ai

            if target_mode == "RTMO - frame":
                should_add_this_ai_pose = True
            elif (target_mode == "RTMO - BBoxes" or target_mode == 'ViTPose - BBoxes') and user_bboxes_for_empty_mode is not None:
                ai_center_x = (bbox_norm_from_ai[0] + bbox_norm_from_ai[2]) / 2
                ai_center_y = (bbox_norm_from_ai[1] + bbox_norm_from_ai[3]) / 2
                for user_ann in user_bboxes_for_empty_mode:
                    # Check if user_ann is "empty" (e.g., few real keypoints set)
                    num_real_kps_user = sum(1 for kp_u in user_ann.keypoints if kp_u[2] in [VISIBILITY_VISIBLE,VISIBILITY_OCCLUDED])
                    if num_real_kps_user > 1: continue # User bbox is not "empty enough"

                    user_b = user_ann.bbox # User's bounding box (normalized)
                    # Check if AI detection center falls within this user's bbox
                    if user_b[0] <= ai_center_x <= user_b[2] and user_b[1] <= ai_center_y <= user_b[3]:
                        # Remove any existing suggestion for this user_ann.id (could be from interpolation)
                        self.suggested_annotations_by_frame[frame_idx] = [
                            s_p for s_p in self.suggested_annotations_by_frame[frame_idx] if s_p.id != user_ann.id
                        ]
                        id_for_this_pose = user_ann.id # Use the ID of the user's annotation
                        bbox_for_this_pose = list(user_ann.bbox) # Use user's bbox
                        should_add_this_ai_pose = True
                        break # Found a match, process this AI pose for this user bbox

            if should_add_this_ai_pose:
                # Avoid adding duplicate suggestions for the same ID (can happen if AI detects multiple times for one user bbox)
                if any(s_p.id == id_for_this_pose for s_p in self.suggested_annotations_by_frame[frame_idx]):
                    if target_mode == "All Detected People": next_ai_id_candidate += 1 # Ensure next new AI ID is unique
                    continue # Skip adding if ID already has a suggestion on this frame

                new_ai_person_ann = PersonAnnotation(person_id=id_for_this_pose, bbox=bbox_for_this_pose, keypoints=normalized_keypoints_data)
                self.suggested_annotations_by_frame[frame_idx].append(new_ai_person_ann)
                num_ai_suggestions_added += 1
                if target_mode == "All Detected People": next_ai_id_candidate += 1 # Increment for the next new AI person

        return num_ai_suggestions_added

    def set_person_done_state(self, person_id: int, is_done: bool):
        """Sets the 'done' state for a person ID globally and saves."""
        if is_done: self.done_person_ids.add(person_id)
        else: self.done_person_ids.discard(person_id)
        self.save_annotations() # Persist the change
        self.done_state_changed.emit(person_id, is_done)

    def is_person_done(self, person_id: int) -> bool:
        """Checks if a person ID is marked as 'done' globally."""
        return person_id in self.done_person_ids