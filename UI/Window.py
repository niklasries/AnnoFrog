# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================

import numpy as np
import cv2
import uuid
from pathlib import Path
import os

# --- OpenGL Imports ---
from OpenGL.GL import *

# --- PyQt6 Imports ---
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import (
    QAction, QImage, QPixmap, QKeyEvent, QColor, QIcon
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QLabel, QWidget,
    QFileDialog, QVBoxLayout, QScrollArea,
    QListWidget, QListWidgetItem, QAbstractItemView, QStatusBar,
    QMenu, QInputDialog, QMessageBox, QPushButton, QButtonGroup, QSpacerItem, QSizePolicy
)

from Annotations.AnnotationHandler import AnnotationHandler
from AI.RTMOManager import RTMOManager
from AI.vitpose_manager import ViTPoseManager
from Annotations.AnnotationMode import AnnotationMode
from Annotations.PersonAnnotation import PersonAnnotation
from UI.InterpolationControlPanel import InterpolationControlPanel
from UI.AIPosePanelWidget import AIPosePanelWidget
from UI.DeleteAnnotationsPanel import DeleteAnnotationsPanel
from UI.ClickableLabel import ClickableLabel
from OGL.OpenGLCanvas import OpenGLCanvas


# Device for AI model inference ('cuda' or 'cpu')
DEVICE = 'cuda'


# Define model paths 
parent = Path(__file__).parent.parent

RTMO_CROWDPOSE_MODEL_PATH = str(parent)+'/res/models/RTMO-body7-crowdpose.onnx' # Example path

print(RTMO_CROWDPOSE_MODEL_PATH)
RTMO_MODEL_INPUT_SIZE = (640, 640)  # Expected input size for the RTMO model

# ==============================================================================
# APPLICATION CONFIGURATION CONSTANTS
# ==============================================================================
THUMBNAIL_HEIGHT = 100  # Height of timeline thumbnails in pixels

# Keypoint visibility states
VISIBILITY_NOT_SET = 0        # Keypoint not yet annotated
VISIBILITY_OCCLUDED = 1       # Keypoint annotated as occluded
VISIBILITY_VISIBLE = 2        # Keypoint annotated as visible
VISIBILITY_SUGGESTED = 3      # Keypoint is an interpolated suggestion
VISIBILITY_AI_SUGGESTED = 4   # Keypoint is an AI-generated suggestion

VITPOSE_12_OUTPUT_TO_ANNOFROG_14_MAP = {
    0: 0,  1: 1,  2: 2,  3: 3,  4: 4,  5: 5,
    6: 6,  7: 7,  8: 8,  9: 9, 10: 10, 11: 11,
}
NUM_KEYPOINTS = 14
VITPOSE_CONF = 0.3 

class Window(QMainWindow):
    """Main application window for the video annotation tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnnoFrog") # (Assuming AiFrog is the project/tool name)
        self.setGeometry(50, 50, 1600, 950) # Default window size and position
        self.setWindowIcon(QIcon('/povid/annotool/AnnoFrog.png'))
        # --- Core Components ---
        self.annotation_handler = AnnotationHandler()
        # self.gl_canvas is initialized in _setup_ui_layout
        self.rtmo_manager = RTMOManager(RTMO_CROWDPOSE_MODEL_PATH, RTMO_MODEL_INPUT_SIZE, device=DEVICE)
        self.vitpose_manager = ViTPoseManager(device=DEVICE)

        # --- Video Data ---
        self.video_frames: list[np.ndarray] = []  # Stores raw BGR frames from the video
        self.video_thumbnails_qimages: list[QImage] = [] # Stores QImage thumbnails for timeline
        self._current_video_loading_task_id: uuid.UUID | None = None # To handle aborted video loads

        # --- UI State ---
        self.globally_hidden_person_ids: set[int] = set() # IDs of persons to hide across all frames

        # --- Setup UI and Connections ---
        self._setup_ui_layout() # Creates and arranges all widgets
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Load a video to begin.")
        self._connect_signals()   # Connects widget signals to handler slots

        self.gl_canvas.setFocus() # Set initial focus to the canvas for keyboard shortcuts
        self.update_mode_button_states(AnnotationMode.IDLE) # Set initial mode

        # Disable AI panel if model isn't ready
        if not self.rtmo_manager.is_ready() and hasattr(self, 'ai_pose_panel'):
            self.ai_pose_panel.setEnabled(False)
            self.statusBar().showMessage("AI Pose Model failed to load. AI features disabled.", 5000)
            print("INFO: AI Pose Panel disabled as RTMO model is not ready.")

    def _create_menu_bar(self):
        """Creates the main menu bar for the application."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        # Define actions for the File menu
        actions_definitions = [
            ("&Load Video...", self.load_video_file, "Ctrl+L"),
            ("&Reset View", self.reset_canvas_view, "Ctrl+R"),
            (None, None, None), # Separator
            ("Clear Annos (Current Frame)", self.clear_annotations_for_current_frame, None),
            ("Save Annos (Force)", self.force_save_annotations, "Ctrl+S"),
            (None, None, None), # Separator
            ("&Exit", self.close, "Ctrl+Q")
        ]

        for text, slot, shortcut in actions_definitions:
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(shortcut)
            file_menu.addAction(action)

    def _setup_ui_layout(self):
        """Creates and arranges all UI widgets within the main window."""
        self._create_menu_bar()

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget) # Main vertical layout for the window

        # --- Top Row: Control Panels, GL Canvas, Persons List ---
        top_row_layout = QHBoxLayout()

        # Left Panel (Combined Controls)
        self.combined_left_panel = QWidget()
        combined_left_layout = QVBoxLayout(self.combined_left_panel)
        combined_left_layout.setContentsMargins(0,0,0,0)
        self.combined_left_panel.setFixedWidth(220) # Fixed width for the left panel

        self.interpolation_panel = InterpolationControlPanel()
        combined_left_layout.addWidget(self.interpolation_panel)

        self.ai_pose_panel = AIPosePanelWidget() # Defined placeholder
        combined_left_layout.addWidget(self.ai_pose_panel)

        self.delete_annotations_panel = DeleteAnnotationsPanel()
        combined_left_layout.addWidget(self.delete_annotations_panel)

        combined_left_layout.addStretch(1) # Push subsequent widgets to bottom

        self.info_label = QLabel("Video Info & Frame Statistics") # For video name, frame count, etc.
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.info_label.setWordWrap(True)
        combined_left_layout.addWidget(self.info_label)

        top_row_layout.addWidget(self.combined_left_panel)

        # GL Canvas (Main display area)
        self.gl_canvas = OpenGLCanvas(self.annotation_handler, self)
        top_row_layout.addWidget(self.gl_canvas, 1) # Canvas takes remaining horizontal space

        # Right Panel (Mode buttons, Persons List)
        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        right_panel_container.setFixedWidth(220)

        # Mode Selection Buttons
        mode_buttons_layout = QHBoxLayout()
        self.select_mode_button = QPushButton("Select/Idle (Esc)")
        self.select_mode_button.setCheckable(True)
        self.select_mode_button.setToolTip("Idle mode. Select existing annotations. (Esc key)")
        self.new_bbox_mode_button = QPushButton("New BBox (B)")
        self.new_bbox_mode_button.setCheckable(True)
        self.new_bbox_mode_button.setToolTip("Create new bounding box. (B key)")

        self.mode_button_group = QButtonGroup(self) # Ensures one button checked at a time
        self.mode_button_group.addButton(self.select_mode_button, AnnotationMode.IDLE.value)
        self.mode_button_group.addButton(self.new_bbox_mode_button, AnnotationMode.CREATING_BBOX_P1.value)
        self.mode_button_group.setExclusive(True)

        mode_buttons_layout.addWidget(self.select_mode_button)
        mode_buttons_layout.addWidget(self.new_bbox_mode_button)
        right_panel_layout.addLayout(mode_buttons_layout)

        right_panel_layout.addWidget(QLabel("Annotated Persons (Current Frame):"))
        self.persons_list_widget = QListWidget()
        self.persons_list_widget.setToolTip("List of persons on current frame.\nDouble-click: Edit person.\nRight-click: More options.")
        right_panel_layout.addWidget(self.persons_list_widget)
        top_row_layout.addWidget(right_panel_container)

        main_layout.addLayout(top_row_layout)

        # --- Bottom Row: Timeline ---
        self.timeline_scroll_area = QScrollArea()
        self.timeline_scroll_area.setWidgetResizable(True) # Important for layout
        self.timeline_scroll_area.setFixedHeight(THUMBNAIL_HEIGHT + 40) # Accommodate thumbnails and scrollbar

        self.timeline_content_container = QWidget() # Content widget for the scroll area
        self.timeline_layout_hbox = QHBoxLayout(self.timeline_content_container) # Horizontal layout for thumbnails
        self.timeline_layout_hbox.setAlignment(Qt.AlignmentFlag.AlignLeft) # Thumbnails align to left
        self.timeline_layout_hbox.setContentsMargins(0,0,0,0) # No extra margins

        self.timeline_scroll_area.setWidget(self.timeline_content_container)
        main_layout.addWidget(self.timeline_scroll_area)

        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        """Connects signals from various widgets to their corresponding slots."""
        # GL Canvas signals
        self.gl_canvas.status_message_changed.connect(self.statusBar().showMessage)
        self.gl_canvas.persons_list_updated.connect(self.refresh_persons_list_display)
        self.gl_canvas.mode_changed_by_canvas.connect(self.update_mode_button_states)
        self.gl_canvas.annotation_action_completed.connect(self.trigger_global_interpolation_update_if_enabled)

        # Annotation Handler signals
        self.annotation_handler.real_annotations_changed.connect(self.trigger_global_interpolation_update_if_enabled)
        self.annotation_handler.done_state_changed.connect(self.handle_person_done_state_changed_externally)

        # Mode buttons
        self.mode_button_group.idClicked.connect(self.handle_mode_button_press) # idClicked gives the ID (enum value)

        # Persons list widget
        self.persons_list_widget.itemDoubleClicked.connect(self.handle_person_list_item_double_clicked)
        self.persons_list_widget.itemClicked.connect(self.handle_person_list_item_single_clicked) # For selection sync
        self.persons_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.persons_list_widget.customContextMenuRequested.connect(self.display_person_context_menu)

        # Interpolation Panel signals
        self.interpolation_panel.interpolate_next_segment_button.clicked.connect(self.handle_generate_next_segment_interpolation)
        self.interpolation_panel.auto_interpolate_checkbox.toggled.connect(self.handle_auto_interpolate_toggled)
        self.interpolation_panel.clear_all_suggestions_button.clicked.connect(self.handle_clear_all_suggestions)
        self.interpolation_panel.accept_current_frame_interp_button.clicked.connect(self.accept_all_interpolated_suggestions_on_current_frame)
        self.interpolation_panel.accept_all_frames_interp_button.clicked.connect(self.handle_accept_all_interpolated_suggestions_all_frames)

        # AI Pose Panel signals (check if exists due to optional RTMO)
        if hasattr(self, 'ai_pose_panel') and self.ai_pose_panel.isEnabled():
            self.ai_pose_panel.run_ai_button.clicked.connect(self.handle_run_ai_pose_estimation)

        # Delete Annotations Panel
        if hasattr(self, 'delete_annotations_panel'):
            self.delete_annotations_panel.delete_requested.connect(self.handle_delete_specified_annotations)

    # --- Mode Management Slots ---
    def handle_mode_button_press(self, mode_enum_value: int):
        """Handles clicks on the mode selection buttons."""
        try:
            new_mode = AnnotationMode(mode_enum_value)
        except ValueError:
            print(f"Warning: Invalid mode value from button: {mode_enum_value}")
            self.update_mode_button_states(AnnotationMode.IDLE) # Fallback
            return

        if new_mode == AnnotationMode.IDLE:
            # When switching to IDLE via button, deselect active person unless it's a suggestion shell
            if self.gl_canvas.active_person and not self.gl_canvas.active_person.is_suggestion_any_type():
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, None) # Deselect
            else:
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, self.gl_canvas.active_person)
        elif new_mode == AnnotationMode.CREATING_BBOX_P1:
            self.gl_canvas.set_annotation_mode(AnnotationMode.CREATING_BBOX_P1)
        self.gl_canvas.setFocus() # Return focus to canvas

    def update_mode_button_states(self, canvas_mode: AnnotationMode):
        """Synchronizes the mode selection buttons with the canvas's current mode."""
        if canvas_mode == AnnotationMode.IDLE and not self.select_mode_button.isChecked():
            self.select_mode_button.setChecked(True)
        elif canvas_mode in [AnnotationMode.CREATING_BBOX_P1, AnnotationMode.CREATING_BBOX_P2] and \
             not self.new_bbox_mode_button.isChecked():
            self.new_bbox_mode_button.setChecked(True)
        elif canvas_mode == AnnotationMode.PLACING_KEYPOINTS and not self.select_mode_button.isChecked():
            # PLACING_KEYPOINTS is conceptually an extension of IDLE (an object is selected)
            self.select_mode_button.setChecked(True)

    # --- Persons List Management ---
    def refresh_persons_list_display(self, frame_idx: int):
        """Updates the list of annotated persons for the given frame index."""
        self.persons_list_widget.blockSignals(True) # Prevent signals during bulk update
        self.persons_list_widget.clear()
        listed_person_ids = set()
        items_to_add = [] # (text, person_id, is_suggestion, is_active)

        # 1. Add Real Annotations
        real_persons_on_frame = sorted(self.annotation_handler.get_annotations_for_frame(frame_idx), key=lambda p: p.id)
        for person in real_persons_on_frame:
            text = f"ID: {person.id}"
            if person.id in self.globally_hidden_person_ids: text += " (Hidden)"
            elif self.annotation_handler.is_person_done(person.id): text += " (Done ✓)"
            elif person.all_keypoints_set(): text += " (✓ KPs Complete)"
            # Check if this person is the active one on the canvas (and it's not a suggestion shell)
            is_active_real = (self.gl_canvas.active_person is person and
                              not (self.gl_canvas.active_person and self.gl_canvas.active_person.is_suggestion_any_type()))
            items_to_add.append((text, person.id, False, is_active_real))
            listed_person_ids.add(person.id)

        # 2. Add Suggested Annotations (if not hidden and no real annotation with same ID)
        suggested_persons_on_frame = sorted(self.annotation_handler.get_suggested_annotations_for_frame(frame_idx), key=lambda p: p.id)
        for sugg_person in suggested_persons_on_frame:
            if sugg_person.id not in self.globally_hidden_person_ids: # Basic hidden check
                sugg_type_str = ""
                is_ai_sugg = sugg_person.has_ai_suggestions()
                is_interp_sugg = sugg_person.has_suggestions() and not is_ai_sugg

                if is_ai_sugg:
                    sugg_type_str = "(AI Sugg. KPs)" # Clarify it's about keypoints
                elif is_interp_sugg:
                    sugg_type_str = "(Interp Sugg)"
                else: # Should not happen if it's in suggested_annotations
                    continue

                text = f"ID: {sugg_person.id} {sugg_type_str}"

                # An AI suggestion (even for an existing ID) can be "active" as a shell
                is_active_suggestion_shell = (self.gl_canvas.active_person is not None and
                                              self.gl_canvas.active_person.id == sugg_person.id and
                                              self.gl_canvas.active_person.is_suggestion_any_type())

                # Add if it's a new ID OR if it's an AI suggestion (we want to see "AI Sugg. KPs" entry)
                # If a real one with same ID is already listed, this adds a second entry for the AI KPs.
                # This might be desired to explicitly show AI is providing keypoints.
                # If you only want one entry per ID, the logic needs to merge display info.
                # For now, let's allow separate listing to make it clear.
                items_to_add.append((text, sugg_person.id, True, is_active_suggestion_shell)) # True for is_suggestion

        # 3. Populate QListWidget
        selected_list_item = None
        for text, person_id, is_suggestion, is_active in items_to_add:
            list_item = QListWidgetItem(text)
            list_item.setData(Qt.ItemDataRole.UserRole, (person_id, is_suggestion)) # Store metadata
            self.persons_list_widget.addItem(list_item)
            if is_active:
                list_item.setSelected(True)
                # Highlight active item
                bg_color = QColor("lightgray") if is_suggestion else Qt.GlobalColor.yellow
                list_item.setBackground(bg_color)
                selected_list_item = list_item
            else:
                list_item.setSelected(False)
                list_item.setBackground(self.persons_list_widget.palette().base()) # Default background

        self.persons_list_widget.blockSignals(False)
        if selected_list_item: # Scroll to the active item
            self.persons_list_widget.scrollToItem(selected_list_item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def handle_person_list_item_single_clicked(self, list_item: QListWidgetItem):
        """Handles single-click on a person in the list to select them on canvas (if not already active)."""
        item_data = list_item.data(Qt.ItemDataRole.UserRole)
        if not item_data: return
        person_id, is_suggestion_item = item_data
        current_frame_idx = self.gl_canvas.current_frame_display_index

        # If person is marked "Done" and it's a real annotation, select for viewing only
        if self.annotation_handler.is_person_done(person_id) and not is_suggestion_item:
            real_person_obj = self.annotation_handler.get_person_by_id_in_frame(current_frame_idx, person_id)
            if real_person_obj and (self.gl_canvas.active_person is not real_person_obj or \
                                   (self.gl_canvas.active_person and self.gl_canvas.active_person.is_suggestion_any_type())):
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, real_person_obj)
            self.gl_canvas.setFocus()
            return

        # Handle suggestion items: select as a "suggestion shell" on canvas
        if is_suggestion_item:
            # Check if a real annotation for this ID now exists (e.g., just accepted)
            real_person_obj = self.annotation_handler.get_person_by_id_in_frame(current_frame_idx, person_id)
            if real_person_obj: # If it became real, select the real one
                if self.gl_canvas.active_person is not real_person_obj or \
                   (self.gl_canvas.active_person and self.gl_canvas.active_person.is_suggestion_any_type()):
                    self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, real_person_obj)
            else: # Still a suggestion, find its object
                suggestion_obj = next((p for p in self.annotation_handler.get_suggested_annotations_for_frame(current_frame_idx)
                                       if p.id == person_id), None)
                if suggestion_obj:
                    # If current active person is not this suggestion shell, update it
                    if self.gl_canvas.active_person is None or \
                       self.gl_canvas.active_person.id != person_id or \
                       not self.gl_canvas.active_person.is_suggestion_any_type():
                        # Create a temporary "shell" PersonAnnotation for the canvas to represent the suggestion
                        shell_person = PersonAnnotation(person_id, bbox=list(suggestion_obj.bbox))
                        shell_person.keypoints = [list(kp) for kp in suggestion_obj.keypoints] # Deep copy
                        self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, shell_person)
                        self.statusBar().showMessage(f"Selected P{person_id} (Suggestion Shell). Double-click or use context menu to accept.")
        else: # Handle real annotation items
            real_person_obj = self.annotation_handler.get_person_by_id_in_frame(current_frame_idx, person_id)
            if real_person_obj and (self.gl_canvas.active_person is not real_person_obj or \
                                   (self.gl_canvas.active_person and self.gl_canvas.active_person.is_suggestion_any_type())):
                # Set this real person as active on the canvas (in IDLE mode)
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, real_person_obj)

        self.gl_canvas.setFocus()

    def handle_person_list_item_double_clicked(self, list_item: QListWidgetItem):
        """Handles double-click on a person in the list to activate editing mode."""
        item_data = list_item.data(Qt.ItemDataRole.UserRole)
        if item_data:
            person_id, is_suggestion_item = item_data
            # If "Done" and real, show message and select for view, but don't edit
            if self.annotation_handler.is_person_done(person_id) and not is_suggestion_item:
                self.statusBar().showMessage(f"P{person_id} is marked Done. Cannot edit.", 3000)
                real_person_obj = self.annotation_handler.get_person_by_id_in_frame(self.gl_canvas.current_frame_display_index, person_id)
                if real_person_obj: self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, real_person_obj)
                return

            # activate_person_for_editing handles accepting suggestion if needed
            if self.gl_canvas.activate_person_for_editing(person_id, is_suggestion_item):
                self.update_mode_button_states(AnnotationMode.PLACING_KEYPOINTS) # Sync mode buttons
        self.gl_canvas.setFocus()

    def display_person_context_menu(self, position: QPoint):
        """Displays a context menu for a person item in the list."""
        list_item = self.persons_list_widget.itemAt(position)
        if not list_item: return
        item_data = list_item.data(Qt.ItemDataRole.UserRole)
        if not item_data: return

        person_id, is_suggestion_item = item_data
        current_frame_idx = self.gl_canvas.current_frame_display_index
        menu = QMenu(self)
        action_taken = False # Flag to refresh list if any action occurs

        if is_suggestion_item:
            suggestion_obj = next((p for p in self.annotation_handler.get_suggested_annotations_for_frame(current_frame_idx)
                                   if p.id == person_id), None)
            if not suggestion_obj: return # Should not happen if item exists

            accept_action = menu.addAction("Accept Suggestion")
            discard_action = menu.addAction("Discard Suggestion")
            menu.addSeparator()
            sugg_type_display = "(AI)" if suggestion_obj.has_ai_suggestions() else "(Interpolated)"
            id_display_action = QAction(f"Suggested ID: {person_id} {sugg_type_display}", self)
            id_display_action.setEnabled(False)
            menu.addAction(id_display_action)

            chosen_action = menu.exec(self.persons_list_widget.mapToGlobal(position))
            action_taken = True # Assume action if menu was shown

            if chosen_action == accept_action:
                is_auto_interpolate = self.interpolation_panel.auto_interpolate_checkbox.isChecked()
                accepted_person = self.annotation_handler.accept_suggestion_for_frame(
                    current_frame_idx, person_id, is_auto_interpolate
                )
                if accepted_person:
                    self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, accepted_person) # Select accepted
                    self.statusBar().showMessage(f"Suggestion for P{person_id} accepted.")
                else:
                    self.statusBar().showMessage(f"Failed to accept suggestion for P{person_id}.")
            elif chosen_action == discard_action:
                self.annotation_handler.discard_suggestion_for_frame(current_frame_idx, person_id)
                self.statusBar().showMessage(f"Suggestion for P{person_id} discarded.")
                # If the discarded suggestion was the active "shell" on canvas, clear active person
                if self.gl_canvas.active_person and self.gl_canvas.active_person.id == person_id and \
                   self.gl_canvas.active_person.is_suggestion_any_type():
                    self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, None)
        else: # Real annotation item
            real_person_obj = self.annotation_handler.get_person_by_id_in_frame(current_frame_idx, person_id)
            if not real_person_obj: return # Should exist

            menu.addAction("Change Person ID Globally...").triggered.connect(
                lambda: self.prompt_change_person_id_globally(person_id)) # Pass only old_id
            menu.addAction("Change Person ID (Current Frame Only)").triggered.connect( 
                lambda: self.prompt_change_person_id_locally(current_frame_idx, person_id))
            menu.addAction("Delete This Person (Current Frame)").triggered.connect(
                lambda: self.delete_person_from_frame(current_frame_idx, person_id))
            menu.addSeparator()

            # Toggle Global Hide
            is_globally_hidden = person_id in self.globally_hidden_person_ids
            hide_text = "Show Globally" if is_globally_hidden else "Hide Globally"
            menu.addAction(hide_text).triggered.connect(
                lambda checked=False, p_id_arg=person_id: self.toggle_person_visibility_globally(p_id_arg))

            # Toggle "Done" State
            is_globally_done = self.annotation_handler.is_person_done(person_id)
            done_text = "Mark as Not Done" if is_globally_done else "Mark as Done"
            menu.addAction(done_text).triggered.connect(
                lambda checked=False, p_id_arg=person_id, current_done_state=is_globally_done: \
                self.annotation_handler.set_person_done_state(p_id_arg, not current_done_state))

            menu.addSeparator()
            id_display_action = QAction(f"Real ID: {person_id}", self)
            id_display_action.setEnabled(False)
            menu.addAction(id_display_action)

            chosen_action = menu.exec(self.persons_list_widget.mapToGlobal(position))
            # Lambdas handle their own logic, so chosen_action might be None.
            # We refresh if *any* interaction with context menu that might change state.
            action_taken = True # Assume any interaction might change state

        if action_taken:
            self.refresh_persons_list_display(current_frame_idx)
            self.gl_canvas.update()

    def toggle_person_visibility_globally(self, person_id: int):
        """Toggles the global visibility state of a person track."""
        if person_id in self.globally_hidden_person_ids:
            self.globally_hidden_person_ids.discard(person_id)
            msg = f"Person P{person_id} will now be Shown Globally."
        else:
            self.globally_hidden_person_ids.add(person_id)
            msg = f"Person P{person_id} will now be Hidden Globally."
            # If the now hidden person was active, deselect them
            if self.gl_canvas.active_person and self.gl_canvas.active_person.id == person_id:
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, None)
        self.statusBar().showMessage(msg)
        self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        self.gl_canvas.update()

    def handle_person_done_state_changed_externally(self, person_id: int, is_done: bool):
        """Handles updates to a person's "done" state triggered by AnnotationHandler."""
        self.statusBar().showMessage(f"Person P{person_id} marked as {'Done' if is_done else 'Not Done'}.")
        if is_done and self.gl_canvas.active_person and self.gl_canvas.active_person.id == person_id:
            # If the person just marked done was active, switch canvas to IDLE mode and deselect
            self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, None)
        self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        self.gl_canvas.update()

    def delete_person_from_frame(self, frame_idx: int, person_id_to_delete: int):
        """Deletes a specific person's annotation from a single frame."""
        current_annotations = self.annotation_handler.get_annotations_for_frame(frame_idx)
        # Create a new list excluding the person to delete
        updated_annotations = [p for p in current_annotations if p.id != person_id_to_delete]

        if len(updated_annotations) < len(current_annotations): # If something was actually removed
            self.annotation_handler.all_annotations_by_frame[frame_idx] = updated_annotations
            if not updated_annotations: # If the list became empty
                del self.annotation_handler.all_annotations_by_frame[frame_idx]
            self.annotation_handler.save_annotations()
            self.statusBar().showMessage(f"Person P{person_id_to_delete} deleted from Frame {frame_idx + 1}.")
            # If the deleted person was active (and not a suggestion shell)
            if self.gl_canvas.active_person and self.gl_canvas.active_person.id == person_id_to_delete and \
               not self.gl_canvas.active_person.is_suggestion_any_type():
                self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE, None) # Deselect
        else:
            self.statusBar().showMessage(f"Person P{person_id_to_delete} not found on Frame {frame_idx + 1} to delete.")

        self.refresh_persons_list_display(frame_idx)
        self.gl_canvas.update()

    def prompt_change_person_id_globally(self, old_person_id: int):
        """Prompts user for a new ID and changes it globally for all frames, handling conflicts by swapping."""
        if not self.video_frames:
             QMessageBox.warning(self, "Error", "No video loaded. Cannot change IDs.")
             return

        # Check if old_person_id actually exists anywhere
        old_id_exists = any(
            any(p.id == old_person_id for p in frame_anns)
            for frame_anns in self.annotation_handler.all_annotations_by_frame.values()
        )
        if not old_id_exists:
            QMessageBox.information(self,"Info",f"Person ID {old_person_id} does not exist in any annotation.")
            return

        new_id, ok = QInputDialog.getInt(self, "Change Person ID Globally",
                                         f"Enter new ID for all instances of Person P{old_person_id}:",
                                         value=old_person_id, min=0, max=99999)
        if not ok: return # User cancelled
        if new_id == old_person_id: return # No change

        num_frames_total = len(self.video_frames)
        new_id_is_used_globally = False
        for f_idx_scan in range(num_frames_total):
            if any(p.id == new_id for p in self.annotation_handler.get_annotations_for_frame(f_idx_scan)):
                new_id_is_used_globally = True
                break

        if new_id_is_used_globally:
            reply = QMessageBox.question(self, "ID Conflict",
                                         f"The ID {new_id} is already in use by another person track.\n\n"
                                         f"Do you want to SWAP all instances of P{old_person_id} with P{new_id} globally?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                                         QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                # Perform swap using a temporary ID
                temp_swap_id = -999 # A temporary ID unlikely to be used
                # Pass 1: old_id -> temp_id
                for f_idx_swap in range(num_frames_total):
                    for p_ann_swap in self.annotation_handler.get_annotations_for_frame(f_idx_swap):
                        if p_ann_swap.id == old_person_id: p_ann_swap.id = temp_swap_id
                # Pass 2: new_id -> old_id
                for f_idx_swap in range(num_frames_total):
                    for p_ann_swap in self.annotation_handler.get_annotations_for_frame(f_idx_swap):
                        if p_ann_swap.id == new_id: p_ann_swap.id = old_person_id
                # Pass 3: temp_id -> new_id
                for f_idx_swap in range(num_frames_total):
                    for p_ann_swap in self.annotation_handler.get_annotations_for_frame(f_idx_swap):
                        if p_ann_swap.id == temp_swap_id: p_ann_swap.id = new_id

                # Swap "done" states
                old_id_was_done = old_person_id in self.annotation_handler.done_person_ids
                new_id_was_done = new_id in self.annotation_handler.done_person_ids
                if old_id_was_done: self.annotation_handler.done_person_ids.remove(old_person_id); self.annotation_handler.done_person_ids.add(new_id)
                else: self.annotation_handler.done_person_ids.discard(new_id) # Ensure new_id not marked done if old wasn't
                if new_id_was_done: self.annotation_handler.done_person_ids.remove(new_id); self.annotation_handler.done_person_ids.add(old_person_id)
                else: self.annotation_handler.done_person_ids.discard(old_person_id) # Ensure old_id not marked done if new wasn't

                self.statusBar().showMessage(f"Globally swapped IDs: P{old_person_id} <-> P{new_id}.")
            elif reply == QMessageBox.StandardButton.No:
                QMessageBox.information(self,"Swap Cancelled", f"ID {new_id} is in use. Choose 'Yes' to swap or pick a different new ID.")
                return
            else: # Cancelled
                self.statusBar().showMessage("ID change cancelled by user.")
                return
        else: # New ID is not used, simple rename
            for f_idx_change in range(num_frames_total):
                for p_ann_change in self.annotation_handler.get_annotations_for_frame(f_idx_change):
                    if p_ann_change.id == old_person_id:
                        p_ann_change.id = new_id
            # Transfer "done" state
            if old_person_id in self.annotation_handler.done_person_ids:
                self.annotation_handler.done_person_ids.remove(old_person_id)
                self.annotation_handler.done_person_ids.add(new_id)
            self.statusBar().showMessage(f"Person ID globally changed from P{old_person_id} to P{new_id}.")

        self.annotation_handler.save_annotations()
        self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        self.gl_canvas.update()


    def prompt_change_person_id_locally(self, frame_idx: int, old_person_id_on_frame: int):
        """Prompts for a new ID and changes it only for the specified person on the current frame."""
        person_to_edit = self.annotation_handler.get_person_by_id_in_frame(frame_idx, old_person_id_on_frame)
        if not person_to_edit:
            QMessageBox.warning(self, "Error", f"Person P{old_person_id_on_frame} not found on Frame {frame_idx + 1}.")
            return

        new_id_local, ok = QInputDialog.getInt(self, "Change Local Person ID",
                                            f"Enter new ID for P{old_person_id_on_frame} on Frame {frame_idx + 1} ONLY:",
                                            value=old_person_id_on_frame, min=0, max=99999)

        if not ok: return
        if new_id_local == old_person_id_on_frame: return # No change

        # Check if the new_id_local is already used by ANOTHER person on THIS frame
        annotations_on_frame = self.annotation_handler.get_annotations_for_frame(frame_idx)
        is_new_id_conflict_on_frame = any(
            p.id == new_id_local and p is not person_to_edit # Check if ID exists AND it's not the same object
            for p in annotations_on_frame
        )

        if is_new_id_conflict_on_frame:
            QMessageBox.warning(self, "ID Conflict on Frame",
                                f"ID {new_id_local} is already used by another person on Frame {frame_idx + 1}. "
                                f"Please choose a different ID for this frame.")
            return

        # If no conflict, change the ID of the specific PersonAnnotation object
        person_to_edit.id = new_id_local
        self.statusBar().showMessage(f"P{old_person_id_on_frame} on Frame {frame_idx + 1} changed to P{new_id_local}.")
        self.annotation_handler.save_annotations() # Save the change
        self.refresh_persons_list_display(frame_idx)
        self.gl_canvas.update()


    # --- Interpolation and Suggestion Handling Slots ---
    def handle_generate_next_segment_interpolation(self):
        """Generates interpolation suggestions for the active person's next segment."""
        if not self.video_frames:
            self.statusBar().showMessage("Load a video first to use interpolation.", 3000)
            return
        if not self.gl_canvas.active_person or \
           self.gl_canvas.active_person.is_suggestion_any_type() or \
           self.annotation_handler.is_person_done(self.gl_canvas.active_person.id):
            self.statusBar().showMessage("Select a real, non-done active person to interpolate their next segment.", 4000)
            return

        start_frame = self.gl_canvas.current_frame_display_index
        person_id_to_interp = self.gl_canvas.active_person.id
        max_gap = self.interpolation_panel.interpolation_gap_spinbox.value()
        interp_type = self.interpolation_panel.interpolation_type_combobox.currentText() # Currently only "Linear"

        self.statusBar().showMessage(f"Generating next suggestion for P{person_id_to_interp} from Frame {start_frame + 1}...", 0)
        QApplication.processEvents() # Allow UI to update status message

        interpolation_result = self.annotation_handler.generate_interpolation_suggestions(
            start_frame, person_id_to_interp, max_gap, interp_type, len(self.video_frames)
        )

        if interpolation_result:
            first_sugg_frame, last_sugg_frame = interpolation_result
            self.statusBar().showMessage(f"P{person_id_to_interp} suggestions generated for Frames {first_sugg_frame + 1} to {last_sugg_frame + 1}.", 5000)
            self.gl_canvas.update() # Redraw canvas to show new suggestions
            self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        else:
            self.statusBar().showMessage(f"No next interpolation segment found for P{person_id_to_interp}. "
                                         f"(No valid next keyframe in range, or segment already interpolated).", 5000)
        self.gl_canvas.setFocus()

    def handle_auto_interpolate_toggled(self, checked: bool):
        """Handles changes to the auto-interpolation checkbox."""
        if checked:
            self.statusBar().showMessage("Auto-interpolation ON. Recalculating suggestions...", 0)
            QApplication.processEvents()
            self.trigger_global_interpolation_update() # Regenerate all suggestions
        else:
            # When turning off, clear all INTERPOLATED suggestions, keep AI ones if any.
            # Or, based on original code, clear all types of suggestions. Let's stick to clear all.
            self.annotation_handler.clear_all_suggestions(preserve_ai_suggestions=False) # Clear ALL suggestions
            self.statusBar().showMessage("Auto-interpolation OFF. All suggestions cleared.", 3000)
            self.gl_canvas.update()
            self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        self.gl_canvas.setFocus()

    def trigger_global_interpolation_update_if_enabled(self):
        """Triggers a global interpolation update only if auto-interpolation is enabled."""
        if hasattr(self, 'interpolation_panel') and self.interpolation_panel.auto_interpolate_checkbox.isChecked():
            self.trigger_global_interpolation_update()

    def trigger_global_interpolation_update(self):
        """Performs interpolation for all valid person tracks across the entire video."""
        if not self.video_frames: return
        print("Starting global interpolation update...")
        QApplication.processEvents() # Allow UI to respond if it's a long process

        # Preserve AI suggestions, clear only interpolated ones before regenerating
        self.annotation_handler.clear_all_suggestions(preserve_ai_suggestions=True)

        max_gap_to_fill = self.interpolation_panel.interpolation_gap_spinbox.value()
        interpolation_type = self.interpolation_panel.interpolation_type_combobox.currentText()
        num_video_frames = len(self.video_frames)

        # Get all unique person IDs that have at least one real keyframe
        all_person_ids_with_keyframes = set()
        for frame_idx_scan in range(num_video_frames):
            for person_ann_scan in self.annotation_handler.get_annotations_for_frame(frame_idx_scan):
                # Check if person has any user-set keypoints on this frame
                if any(kp[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED] for kp in person_ann_scan.keypoints):
                    all_person_ids_with_keyframes.add(person_ann_scan.id)

        total_suggestions_generated_count = 0
        for person_id_to_process in sorted(list(all_person_ids_with_keyframes)):
            if self.annotation_handler.is_person_done(person_id_to_process):
                continue # Skip "done" persons for interpolation

            # Find all frame indices where this person has a real keyframe
            keyframes_for_this_id = []
            for f_idx_kf_scan in range(num_video_frames):
                person_at_frame = self.annotation_handler.get_person_by_id_in_frame(f_idx_kf_scan, person_id_to_process)
                if person_at_frame and any(kp[2] in [VISIBILITY_VISIBLE, VISIBILITY_OCCLUDED] for kp in person_at_frame.keypoints):
                    keyframes_for_this_id.append(f_idx_kf_scan)

            if len(keyframes_for_this_id) < 2: continue # Need at least two keyframes to interpolate between

            # Iterate through pairs of keyframes for this person
            for i_kf_pair_idx in range(len(keyframes_for_this_id) - 1):
                start_kf_idx = keyframes_for_this_id[i_kf_pair_idx]
                end_kf_idx = keyframes_for_this_id[i_kf_pair_idx + 1]
                num_frames_in_gap = end_kf_idx - start_kf_idx - 1

                if 0 < num_frames_in_gap < max_gap_to_fill : # Gap exists and is within user-defined max_gap
                    # Max search distance for generate_interpolation_suggestions should be to the end_kf_idx
                    search_distance_for_end_kf = end_kf_idx - start_kf_idx
                    QApplication.processEvents() # Keep UI responsive during loops
                    interp_result_tuple = self.annotation_handler.generate_interpolation_suggestions(
                        start_kf_idx, person_id_to_process, search_distance_for_end_kf,
                        interpolation_type, num_video_frames
                    )
                    if interp_result_tuple and interp_result_tuple[1] >= interp_result_tuple[0]:
                        total_suggestions_generated_count += (interp_result_tuple[1] - interp_result_tuple[0] + 1)

        if total_suggestions_generated_count > 0:
            self.statusBar().showMessage(f"Auto-interpolation: Generated {total_suggestions_generated_count} suggested keypoint sets globally.", 5000)
        elif self.interpolation_panel.auto_interpolate_checkbox.isChecked(): # Only show if auto-interp is on
            self.statusBar().showMessage("Auto-interpolation: No new suggestions to generate (gaps covered or outside max_gap).", 3000)

        self.gl_canvas.update()
        self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        print("Global interpolation update finished.")

    def handle_clear_all_suggestions(self):
        """Clears all types of suggestions (interpolated and AI) from all frames."""
        self.annotation_handler.clear_all_suggestions(preserve_ai_suggestions=False)
        self.statusBar().showMessage("All suggestions (interpolated and AI) cleared from all frames.", 3000)
        self.gl_canvas.update()
        self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
        self.gl_canvas.setFocus()

    def accept_all_interpolated_suggestions_on_current_frame(self):
        """Accepts all PURELY interpolated (non-AI) suggestions on the current frame."""
        current_frame_idx = self.gl_canvas.current_frame_display_index
        if current_frame_idx < 0 or not self.video_frames:
            self.statusBar().showMessage("No frame loaded to accept suggestions on.", 2000)
            return

        # Get a copy of suggestions list as it will be modified during iteration
        suggestions_on_frame = list(self.annotation_handler.get_suggested_annotations_for_frame(current_frame_idx))
        accepted_count = 0
        is_auto_interpolate_mode_on = self.interpolation_panel.auto_interpolate_checkbox.isChecked()

        for suggested_person in suggestions_on_frame:
            # Accept only if it's purely interpolated (has VISIBILITY_SUGGESTED)
            # AND does NOT have any AI suggestions (has VISIBILITY_AI_SUGGESTED)
            is_purely_interpolated = (any(kp[2] == VISIBILITY_SUGGESTED for kp in suggested_person.keypoints) and
                                      not any(kp[2] == VISIBILITY_AI_SUGGESTED for kp in suggested_person.keypoints))
            if is_purely_interpolated:
                accepted_person_obj = self.annotation_handler.accept_suggestion_for_frame(
                    current_frame_idx, suggested_person.id, is_auto_interpolate_mode_on
                )
                if accepted_person_obj:
                    accepted_count += 1

        if accepted_count > 0:
            self.statusBar().showMessage(f"Accepted {accepted_count} interpolated suggestion(s) on Frame {current_frame_idx + 1}.", 3000)
            self.refresh_persons_list_display(current_frame_idx)
            self.gl_canvas.update()
        else:
            self.statusBar().showMessage(f"No purely interpolated suggestions found to accept on Frame {current_frame_idx + 1}.", 2000)
        self.gl_canvas.setFocus()

    def handle_accept_all_interpolated_suggestions_all_frames(self):
        """Accepts all PURELY interpolated (non-AI) suggestions across ALL frames."""
        if not self.video_frames:
            self.statusBar().showMessage("No video loaded to accept suggestions.", 2000)
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Mass Acceptance")
        msg_box.setText("Are you sure you want to accept all PURELY INTERPOLATED suggestions across ALL frames?\n"
                        "AI-generated suggestions will NOT be affected by this action.\n"
                        "This action cannot be easily undone.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        if msg_box.exec() == QMessageBox.StandardButton.No:
            return

        self.statusBar().showMessage("Accepting all interpolated suggestions across all frames... This may take a moment.", 0)
        QApplication.processEvents()

        total_accepted_count = 0
        is_auto_interpolate_mode_on = self.interpolation_panel.auto_interpolate_checkbox.isChecked()
        # Iterate over a copy of frame indices as the underlying dict might change
        frames_with_suggestions_indices = list(self.annotation_handler.suggested_annotations_by_frame.keys())

        for frame_idx_process in frames_with_suggestions_indices:
            # Get a copy of this frame's suggestions list
            suggestions_on_this_frame = list(self.annotation_handler.get_suggested_annotations_for_frame(frame_idx_process))
            for suggested_person in suggestions_on_this_frame:
                is_purely_interpolated = (any(kp[2] == VISIBILITY_SUGGESTED for kp in suggested_person.keypoints) and
                                          not any(kp[2] == VISIBILITY_AI_SUGGESTED for kp in suggested_person.keypoints))
                if is_purely_interpolated:
                    accepted_person_obj = self.annotation_handler.accept_suggestion_for_frame(
                        frame_idx_process, suggested_person.id, is_auto_interpolate_mode_on
                    )
                    if accepted_person_obj:
                        total_accepted_count += 1
            # Provide progress update periodically
            if frame_idx_process % 20 == 0: # Update status bar every 20 frames
                QApplication.processEvents()
                self.statusBar().showMessage(f"Processing... Accepted {total_accepted_count} suggestions so far.")

        # Final save is handled within accept_suggestion_for_frame calls.
        if total_accepted_count > 0:
            self.statusBar().showMessage(f"Accepted {total_accepted_count} interpolated suggestion(s) across all frames.", 5000)
            self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
            self.gl_canvas.update()
        else:
            self.statusBar().showMessage("No purely interpolated suggestions found to accept across all frames.", 3000)
        self.gl_canvas.setFocus()


    # --- AI Pose Estimation Slots ---
    def handle_run_ai_pose_estimation(self):
        if not self.video_frames or self.gl_canvas.current_frame_display_index < 0:
            self.statusBar().showMessage("No video/frame loaded.", 3000)
            return

        current_frame_idx = self.gl_canvas.current_frame_display_index
        frame_bgr_data = self.video_frames[current_frame_idx]
        frame_height, frame_width = frame_bgr_data.shape[:2]

        selected_mode_text = self.ai_pose_panel.detection_mode_combobox.currentText() # From your AIPosePanelWidget
        user_bboxes_for_target_mode = self.annotation_handler.get_annotations_for_frame(current_frame_idx)
        kp_thresh = self.ai_pose_panel.kp_confidence_spinbox.value()
        
        self.statusBar().showMessage(f"Running AI ({selected_mode_text}) on F{current_frame_idx + 1}...", 0)
        QApplication.processEvents()

        if "ViTPose - BBoxes" == selected_mode_text:
            # --- ViTPose Specific Logic ---
            if not self.vitpose_manager or not self.vitpose_manager.is_ready():
                self.statusBar().showMessage("ViTPose Model not ready.", 3000)
                return

            user_annotations_on_frame = self.annotation_handler.get_annotations_for_frame(current_frame_idx)
            active_user_annotations = [
                ann for ann in user_annotations_on_frame if not (ann.id in self.globally_hidden_person_ids) and \
                                                            not self.annotation_handler.is_person_done(ann.id) and \
                                                            ann.bbox != [0,0,0,0]
            ]
            if not active_user_annotations:
                self.statusBar().showMessage("ViTPose: No active, non-done user bboxes on this frame.", 3000)
                return

            bboxes_for_vitpose_pixel_xywh = []
            ids_and_src_bbox_info = [] 

            for user_ann in active_user_annotations:
                norm_bbox = user_ann.bbox
                x_min_px = norm_bbox[0] * frame_width
                y_min_px = norm_bbox[1] * frame_height
                x_max_px = norm_bbox[2] * frame_width
                y_max_px = norm_bbox[3] * frame_height
                w_px = x_max_px - x_min_px
                h_px = y_max_px - y_min_px
                if w_px <= 0 or h_px <= 0: continue
                bboxes_for_vitpose_pixel_xywh.append([x_min_px, y_min_px, w_px, h_px])
                ids_and_src_bbox_info.append({'id': user_ann.id, 'bbox_norm': list(user_ann.bbox)})
            
            if not bboxes_for_vitpose_pixel_xywh:
                self.statusBar().showMessage("ViTPose: No valid bboxes after conversion.", 3000)
                return
            
            vitpose_raw_results = self.vitpose_manager.predict_poses_from_bboxes(frame_bgr_data, bboxes_for_vitpose_pixel_xywh)

            if vitpose_raw_results is None: self.statusBar().showMessage("ViTPose prediction failed.", 3000); return
            if not vitpose_raw_results: self.statusBar().showMessage("ViTPose: No poses detected.", 3000); return
            if len(vitpose_raw_results) != len(ids_and_src_bbox_info):
                 self.statusBar().showMessage("ViTPose: Mismatch input bboxes and results count.", 3000); return

            # Prepare data for AnnotationHandler
            ai_poses_for_handler_vitpose = []
            for i, vit_res_dict in enumerate(vitpose_raw_results):
                person_id = ids_and_src_bbox_info[i]['id']
                original_user_bbox_norm = ids_and_src_bbox_info[i]['bbox_norm']
                
                keypoints_12_pixel = vit_res_dict['keypoints'] 
                scores_12 = vit_res_dict['scores']

                # Transform ViTPose 12-keypoint output to AnnoFrog 14-keypoint structure
                annofrog_kps_pixel = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
                annofrog_scores = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

                for vit_idx_12, af_idx_14 in VITPOSE_12_OUTPUT_TO_ANNOFROG_14_MAP.items():
                    if vit_idx_12 < keypoints_12_pixel.shape[0]:
                        annofrog_kps_pixel[af_idx_14] = keypoints_12_pixel[vit_idx_12]
                        annofrog_scores[af_idx_14] = scores_12[vit_idx_12]
                
                # Head and Neck will have 0 scores and (0,0) pixels unless estimated above
                # (You mentioned not to estimate them, so they'll be VISIBILITY_NOT_SET by handler)

                ai_poses_for_handler_vitpose.append({
                    'keypoints': annofrog_kps_pixel,
                    'scores': annofrog_scores,
                    'person_id': person_id,          # Crucial: provide the target person ID
                    'bbox_norm': original_user_bbox_norm, # And the original bbox
                    'source_model_hint': 'vitpose' # For followup logic
                })
            
            if ai_poses_for_handler_vitpose:
                # Call your existing AnnotationHandler method.
                # It needs to be able to use 'person_id' and 'bbox_norm' if present in the dict.
                num_suggestions_added = self.annotation_handler.add_ai_pose_suggestions_to_frame(
                    current_frame_idx,
                    ai_poses_for_handler_vitpose, # List of dicts, each for one person
                    kp_thresh,
                    selected_mode_text, # This is now implicitly handled by providing person_id/bbox_norm
                    user_bboxes_for_target_mode, # Not needed if we pass person_id/bbox_norm directly
                    frame_width=frame_width,
                    frame_height=frame_height
                )
                if num_suggestions_added > 0:
                    self.statusBar().showMessage(f"ViTPose: Added/Updated {num_suggestions_added} suggestions.", 3000)
                    self.gl_canvas.update()
                    self.refresh_persons_list_display(current_frame_idx)
                else:
                    self.statusBar().showMessage("ViTPose: No new suggestions met criteria.", 3000)
            else:
                self.statusBar().showMessage("ViTPose: No poses processed.", 3000)

        elif "RTMO - BBoxes" == selected_mode_text or \
             "RTMO - frame" == selected_mode_text:
            
            if not self.rtmo_manager or not self.rtmo_manager.is_ready():
                self.statusBar().showMessage("RTMO Model is not ready or failed to load.", 3000) 
                return

            ai_poses_result_rtmo = self.rtmo_manager.predict_poses(frame_bgr_data)

            if ai_poses_result_rtmo is None:
                self.statusBar().showMessage("RTMO Pose Estimation failed.", 3000)
                return
            if not ai_poses_result_rtmo:
                self.statusBar().showMessage("RTMO: No poses detected on this frame.", 3000)
                return

            # Add source hint to RTMO results
            for res in ai_poses_result_rtmo:
                res['source_model_hint'] = 'rtmo'
                if "RTMO - Only for Empty User BBoxes" == selected_mode_text:
                    res['source_model_hint'] = 'rtmo_empty_attempt'


            user_bboxes_for_rtmo_empty_mode = None
            if "RTMO - BBoxes" == selected_mode_text:
                #user_bboxes_for_rtmo_empty_mode = self.annotation_handler.get_annotations_for_frame(current_frame_idx)
                
                num_suggestions_added = self.annotation_handler.add_ai_pose_suggestions_to_frame(
                    current_frame_idx, ai_poses_result_rtmo, kp_thresh,
                    selected_mode_text, # Pass the mode string
                    user_bboxes_for_target_mode, # Pass the user bboxes
                    frame_width=frame_width,
                    frame_height=frame_height
                )

            else: # "RTMO - All Detected People"
                 num_suggestions_added = self.annotation_handler.add_ai_pose_suggestions_to_frame(
                    current_frame_idx, ai_poses_result_rtmo, kp_thresh,
                    selected_mode_text, # "RTMO - All Detected People"
                    None, # No specific user_bboxes for this mode
                    frame_width=frame_width,
                    frame_height=frame_height
                )


            if num_suggestions_added > 0:
                self.statusBar().showMessage(f"RTMO: Added {num_suggestions_added} new pose suggestions.", 4000) # Corrected duration
                self.gl_canvas.update()
                self.refresh_persons_list_display(current_frame_idx)
            else:
                self.statusBar().showMessage("RTMO: No new suggestions met criteria.", 3000)
        else:
            self.statusBar().showMessage(f"Unknown AI detection mode: {selected_mode_text}", 3000)
            return

        self.gl_canvas.setFocus()

    # --- Video Loading and Timeline Management ---
    def load_video_file(self):
        """Opens a file dialog to load a video, then processes it frame by frame."""
        # Generate a unique ID for this loading task to handle potential interruptions
        current_task_id = uuid.uuid4()
        self._current_video_loading_task_id = current_task_id

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not filepath: # User cancelled dialog
            if self._current_video_loading_task_id == current_task_id:
                self._current_video_loading_task_id = None # Clear task ID if this was the one cancelled
            return

        print(f"Starting video load (Task ID: {current_task_id}): {filepath}")
        self.statusBar().showMessage(f"Loading video: {filepath.split('/')[-1]}...")

        # --- Reset application state for new video ---
        self.video_frames.clear()
        self.video_thumbnails_qimages.clear()
        self.annotation_handler.all_annotations_by_frame.clear()
        self.annotation_handler.current_json_path = None # Will be set by load_annotations_for_video
        self.annotation_handler.clear_all_suggestions(preserve_ai_suggestions=False)
        self.annotation_handler.done_person_ids.clear()
        self.globally_hidden_person_ids.clear()

        if self.gl_canvas.active_person: self.gl_canvas.active_person = None
        self.gl_canvas.set_annotation_mode(AnnotationMode.IDLE)
        self.gl_canvas.current_frame_display_index = 0
        self.gl_canvas.update_video_frame(None, 0) # Clear canvas

        # Clear existing timeline thumbnails
        timeline_item = self.timeline_layout_hbox.takeAt(0)
        while timeline_item:
            if timeline_item.widget():
                timeline_item.widget().deleteLater() # Schedule for deletion
            del timeline_item # Remove item from layout
            timeline_item = self.timeline_layout_hbox.takeAt(0)

        # Load associated annotations (or start fresh if none)
        self.annotation_handler.load_annotations_for_video(filepath)

        # --- Load video frames using OpenCV ---
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            self.statusBar().showMessage(f"Error opening video file: {filepath}", 5000)
            if self._current_video_loading_task_id == current_task_id: self._current_video_loading_task_id = None
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info_label.setText(f"Loading Video (Task {str(current_task_id)[:4]}):\n{filepath.split('/')[-1]}\n0 / {total_frames if total_frames > 0 else 'Unknown'}")
        QApplication.processEvents() # Update UI

        processed_frame_count = 0
        # Progress update interval (e.g., every 5% of total frames)
        progress_update_interval = max(1, total_frames // 20 if total_frames > 0 else 1)

        try:
            while True:
                # Check if this loading task has been superseded by a new one
                if self._current_video_loading_task_id != current_task_id:
                    print(f"Video load task {current_task_id} aborted by new load request.")
                    break # Exit loop if task ID changed

                ret, frame_bgr = cap.read()
                if not ret: break # End of video or error

                self.video_frames.append(frame_bgr.copy()) # Store BGR frame

                # Create thumbnail
                h, w = frame_bgr.shape[:2]
                if h == 0: continue # Skip if frame height is zero (corrupt frame?)
                thumb_width = max(1, int(w * (THUMBNAIL_HEIGHT / h) if h > 0 else THUMBNAIL_HEIGHT))
                thumbnail_bgr = cv2.resize(frame_bgr, (thumb_width, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA)
                thumbnail_rgb = cv2.cvtColor(thumbnail_bgr, cv2.COLOR_BGR2RGB)
                qimage = QImage(thumbnail_rgb.data, thumb_width, THUMBNAIL_HEIGHT,
                                3 * thumb_width, QImage.Format.Format_RGB888).copy() # Crucial to copy()
                self.video_thumbnails_qimages.append(qimage)

                processed_frame_count += 1
                if processed_frame_count % progress_update_interval == 0:
                    if self._current_video_loading_task_id != current_task_id: # Re-check before UI update
                        print(f"Video load task {current_task_id} aborted during UI update phase.")
                        break
                    self.info_label.setText(f"Loading Video (Task {str(current_task_id)[:4]})...\n"
                                            f"{processed_frame_count} / {total_frames if total_frames > 0 else 'Unknown'}")
                    QApplication.processEvents() # Keep UI responsive
        finally:
            cap.release()

        # Finalize loading only if this task was not aborted
        if self._current_video_loading_task_id == current_task_id:
            self.info_label.setText(f"Loaded: {filepath.split('/')[-1]}\nFrames: {len(self.video_frames)}")
            self.statusBar().showMessage(f"Video loaded: {len(self.video_frames)} frames. Annotations processed.", 5000)

            self._populate_timeline_widget()
            if self.video_frames:
                self.display_frame_by_index(0) # Display first frame

            self.reset_canvas_view()
            self.gl_canvas.setFocus()
            self._current_video_loading_task_id = None # Mark task as completed
            print(f"Video load task {current_task_id} completed successfully.")
            # Trigger initial global interpolation if auto-interpolate is on
            self.trigger_global_interpolation_update_if_enabled()
        else:
            # This task was aborted, but its loop finished.
            # A new task is likely in progress or completed.
            print(f"Video load task {current_task_id} finished, but was previously aborted. No further action.")

    def display_frame_by_index(self, frame_idx_to_display: int):
        """Displays the video frame at the given index on the canvas and updates related UI."""
        if not (0 <= frame_idx_to_display < len(self.video_frames)):
            print(f"Error: Frame index {frame_idx_to_display} out of bounds (0-{len(self.video_frames)-1}).")
            return

        # Preserve active person and mode if possible when changing frames
        preserved_active_person_id: int | None = None
        current_mode_before_change = self.gl_canvas.current_mode
        is_active_person_a_suggestion_shell = False

        if self.gl_canvas.active_person:
            preserved_active_person_id = self.gl_canvas.active_person.id
            is_active_person_a_suggestion_shell = self.gl_canvas.active_person.is_suggestion_any_type()

        self.gl_canvas.active_person = None # Temporarily clear for mode setting

        # Update GLCanvas with the new frame
        self.current_frame_display_idx = frame_idx_to_display # Store current index at Window level too
        self.gl_canvas.update_video_frame(self.video_frames[frame_idx_to_display], frame_idx_to_display)

        # Refresh the persons list for the new frame
        self.refresh_persons_list_display(frame_idx_to_display)

        # Try to restore active person on the new frame
        final_active_person_for_new_frame: PersonAnnotation.PersonAnnotation | None = None
        if preserved_active_person_id is not None:
            # Check for real annotation first
            real_person_on_new_frame = self.annotation_handler.get_person_by_id_in_frame(
                frame_idx_to_display, preserved_active_person_id
            )
            if real_person_on_new_frame:
                final_active_person_for_new_frame = real_person_on_new_frame
            elif is_active_person_a_suggestion_shell: # If previous was a shell, try to find suggestion on new frame
                suggested_person_on_new_frame = next(
                    (p for p in self.annotation_handler.get_suggested_annotations_for_frame(frame_idx_to_display)
                     if p.id == preserved_active_person_id), None
                )
                if suggested_person_on_new_frame:
                    # Create a new shell for the new frame's suggestion
                    shell = PersonAnnotation(preserved_active_person_id, bbox=list(suggested_person_on_new_frame.bbox))
                    shell.keypoints = [list(kp) for kp in suggested_person_on_new_frame.keypoints]
                    final_active_person_for_new_frame = shell

        # Determine the mode for the new frame
        new_mode_for_canvas = AnnotationMode.IDLE
        if current_mode_before_change == AnnotationMode.PLACING_KEYPOINTS and \
           final_active_person_for_new_frame and \
           not final_active_person_for_new_frame.is_suggestion_any_type() and \
           not self.annotation_handler.is_person_done(final_active_person_for_new_frame.id):
            # If was placing keypoints and the active person (now real) is editable on new frame, continue
            new_mode_for_canvas = AnnotationMode.PLACING_KEYPOINTS

        self.gl_canvas.set_annotation_mode(new_mode_for_canvas, final_active_person_for_new_frame)

        # Update timeline UI
        self.center_timeline_on_frame(frame_idx_to_display)
        self._highlight_timeline_frame(frame_idx_to_display)
        self.gl_canvas.setFocus()

    def center_timeline_on_frame(self, frame_idx: int):
        """Scrolls the timeline to center the thumbnail for the given frame index."""
        if not (0 <= frame_idx < self.timeline_layout_hbox.count()): return

        target_label_item = self.timeline_layout_hbox.itemAt(frame_idx)
        if target_label_item and target_label_item.widget():
            target_label_widget = target_label_item.widget()
            scroll_area_viewport_width = self.timeline_scroll_area.viewport().width()
            # Calculate the target X position to center the label
            target_label_center_x = target_label_widget.pos().x() + target_label_widget.width() / 2
            new_scroll_value = int(target_label_center_x - scroll_area_viewport_width / 2)

            h_scrollbar = self.timeline_scroll_area.horizontalScrollBar()
            # Clamp scroll value to min/max of scrollbar
            new_scroll_value = max(h_scrollbar.minimum(), min(new_scroll_value, h_scrollbar.maximum()))
            h_scrollbar.setValue(new_scroll_value)

    def _highlight_timeline_frame(self, current_frame_idx: int):
        """Highlights the current frame's thumbnail in the timeline and unhighlights others."""
        for i in range(self.timeline_layout_hbox.count()):
            item = self.timeline_layout_hbox.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ClickableLabel):
                label_widget: ClickableLabel = item.widget() # type: ignore
                if label_widget.frame_id == current_frame_idx:
                    label_widget.setStyleSheet("border: 2px solid lightgreen; padding: 1px;") # Highlight
                else:
                    label_widget.setStyleSheet("border: none; padding: 3px;") # Default

    def _populate_timeline_widget(self):
        """Populates the timeline scroll area with clickable frame thumbnails."""
        # Clear existing items first (should have been done in load_video_file, but good practice)
        item = self.timeline_layout_hbox.takeAt(0)
        while item:
            if item.widget(): item.widget().deleteLater()
            del item
            item = self.timeline_layout_hbox.takeAt(0)

        # Add new thumbnails
        for i, qimage_thumbnail in enumerate(self.video_thumbnails_qimages):
            thumbnail_label = ClickableLabel(i, QPixmap.fromImage(qimage_thumbnail), self.timeline_content_container)
            thumbnail_label.clicked.connect(self.display_frame_by_index) # Connect click to frame display
            self.timeline_layout_hbox.addWidget(thumbnail_label)

        # Add a spacer item to push thumbnails to the left if they don't fill the area
        self.timeline_layout_hbox.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        if self.video_thumbnails_qimages:
            current_display_idx = self.gl_canvas.current_frame_display_index if hasattr(self.gl_canvas, 'current_frame_display_index') else 0
            self._highlight_timeline_frame(current_display_idx)

    # --- Miscellaneous Actions ---
    def reset_canvas_view(self):
        """Resets the GL canvas zoom and pan to default."""
        if hasattr(self, 'gl_canvas'):
            self.gl_canvas.view_scale = 1.0
            self.gl_canvas.view_translation = np.array([0.0, 0.0], dtype=np.float32)
            self.gl_canvas.update()
            self.statusBar().showMessage("Canvas view reset.", 2000)

    def clear_annotations_for_current_frame(self):
        """Clears all annotations (real and suggested) for the current frame."""
        if not self.video_frames:
            self.statusBar().showMessage("No video loaded.", 2000)
            return
        self.gl_canvas.clear_all_annotations_on_current_frame()
        # Status message is handled by gl_canvas method

    def force_save_annotations(self):
        """Forces a save of the current annotations to file."""
        if not self.annotation_handler.current_json_path:
            self.statusBar().showMessage("No annotation file context. Load a video first.", 3000)
            return
        self.annotation_handler.save_annotations()
        self.statusBar().showMessage("Annotations saved successfully.", 3000)

    def handle_delete_specified_annotations(self, target_ids: list[int], frame_specs: list):
        """Deletes annotations based on specified Person IDs and frame ranges/specs."""
        if not self.video_frames:
            self.statusBar().showMessage("No video loaded. Cannot delete annotations.", 3000)
            return

        frames_to_process_indices = set() # Set of 0-indexed frame indices
        max_frame_idx = len(self.video_frames) - 1

        if frame_specs == ["all"]: # Special keyword "all"
            frames_to_process_indices.update(range(len(self.video_frames)))
        else:
            for spec_start_0idx, spec_end_0idx in frame_specs: # These are already 0-indexed from panel
                start_f = max(0, spec_start_0idx)
                end_f = min(max_frame_idx, spec_end_0idx)
                if start_f <= end_f: # Valid range
                    frames_to_process_indices.update(range(start_f, end_f + 1))

        if not frames_to_process_indices:
            self.statusBar().showMessage("No valid frames selected for deletion.", 3000)
            return

        confirm_msg = (f"Are you sure you want to delete annotations for Person IDs: {target_ids} "
                       f"on frames: {self.format_frame_ranges_for_display(list(frames_to_process_indices))}?\n\n"
                       f"This action CANNOT be undone.")
        reply = QMessageBox.question(self, "Confirm Deletion", confirm_msg,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No) # Default to No
        if reply == QMessageBox.StandardButton.No:
            self.statusBar().showMessage("Deletion cancelled.", 2000)
            return

        deleted_annotations_count = 0
        for frame_idx_to_delete_from in sorted(list(frames_to_process_indices)):
            annotations_on_this_frame = self.annotation_handler.get_annotations_for_frame(frame_idx_to_delete_from)
            if not annotations_on_this_frame: continue # Skip if frame has no annotations

            initial_count_on_frame = len(annotations_on_this_frame)
            # Filter out annotations matching the target IDs
            self.annotation_handler.all_annotations_by_frame[frame_idx_to_delete_from] = [
                ann for ann in annotations_on_this_frame if ann.id not in target_ids
            ]
            deleted_annotations_count += (initial_count_on_frame - len(self.annotation_handler.all_annotations_by_frame[frame_idx_to_delete_from]))

            # If frame becomes empty of annotations, remove its entry
            if not self.annotation_handler.all_annotations_by_frame[frame_idx_to_delete_from]:
                del self.annotation_handler.all_annotations_by_frame[frame_idx_to_delete_from]

        if deleted_annotations_count > 0:
            self.annotation_handler.save_annotations() # Save changes
            self.statusBar().showMessage(f"Successfully deleted {deleted_annotations_count} annotation instance(s).", 4000)
            self.refresh_persons_list_display(self.gl_canvas.current_frame_display_index)
            self.gl_canvas.update()
        else:
            self.statusBar().showMessage("No matching annotations found to delete based on criteria.", 3000)

    def format_frame_ranges_for_display(self, frame_indices_zero_based: list[int]) -> str:
        """Converts a list of 0-indexed frame numbers to a compact string (e.g., "1-3, 5, 7-8")."""
        if not frame_indices_zero_based: return "None"
        frame_indices_zero_based.sort()
        ranges = []
        start_of_range = -1
        for i in range(len(frame_indices_zero_based)):
            current_frame_one_based = frame_indices_zero_based[i] + 1 # Convert to 1-indexed for display
            if start_of_range == -1:
                start_of_range = current_frame_one_based

            # If it's the last frame or the next frame is not consecutive
            if i + 1 == len(frame_indices_zero_based) or \
               frame_indices_zero_based[i+1] != frame_indices_zero_based[i] + 1:
                if start_of_range == current_frame_one_based: # Single frame in range
                    ranges.append(str(start_of_range))
                else: # Multiple frames in range
                    ranges.append(f"{start_of_range}-{current_frame_one_based}")
                start_of_range = -1 # Reset for next potential range
        return ", ".join(ranges)

    # --- Global Key Press Event Handler ---
    def keyPressEvent(self, event: QKeyEvent):
        """Handles global keyboard shortcuts for the main window."""
        # Allow loading video even if no video is currently loaded
        if not self.video_frames and not (event.key() == Qt.Key.Key_L and event.modifiers() == Qt.KeyboardModifier.ControlModifier):
            super().keyPressEvent(event) # Pass to Qt if no video and not Ctrl+L
            return

        current_idx = self.gl_canvas.current_frame_display_index
        num_frames = len(self.video_frames) if self.video_frames else 0

        # Frame navigation
        if event.key() == Qt.Key.Key_A: # Previous frame
            if num_frames > 0 and current_idx > 0:
                self.display_frame_by_index(current_idx - 1)
            event.accept()
        elif event.key() == Qt.Key.Key_D: # Next frame
            if num_frames > 0 and current_idx < num_frames - 1:
                self.display_frame_by_index(current_idx + 1)
            event.accept()
        elif event.key() == Qt.Key.Key_V: # Run AI
             if self.ai_pose_panel.isEnabled(): self.handle_run_ai_pose_estimation()
             else: self.statusBar().showMessage("AI features are disabled.", 2000)
             event.accept()
        elif event.key() == Qt.Key.Key_Space: # Accept suggestions on current frame
            if self.video_frames and current_idx >= 0:
                self.accept_all_interpolated_suggestions_on_current_frame()
            event.accept()
        elif event.key() == Qt.Key.Key_J: # Jump to frame
            if self.video_frames:
                max_f_display = len(self.video_frames)
                frame_num_str, ok = QInputDialog.getText(self, "Jump to Frame",
                                                         f"Enter frame number (1 - {max_f_display}):")
                if ok and frame_num_str.isdigit():
                    frame_num_one_indexed = int(frame_num_str)
                    frame_num_zero_indexed = frame_num_one_indexed - 1
                    if 0 <= frame_num_zero_indexed < max_f_display:
                        self.display_frame_by_index(frame_num_zero_indexed)
                    else:
                        QMessageBox.warning(self, "Invalid Frame",
                                            f"Frame number must be between 1 and {max_f_display}.")
            event.accept()
        else:
            # Important: Pass unhandled events to superclass (allows GLCanvas to get key events too)
            super().keyPressEvent(event)

    def _is_person_globally_hidden(self, person_id: int) -> bool:
        """Checks if a person ID is in the window's globally_hidden_person_ids set."""
        return person_id in self.globally_hidden_person_ids