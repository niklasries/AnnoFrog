# ==============================================================================
# UI PANEL WIDGETS
# ==============================================================================

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QWidget, QVBoxLayout, QGroupBox, QSpinBox, QComboBox, QCheckBox, QPushButton)

class InterpolationControlPanel(QWidget):
    """Panel containing controls for interpolation tools."""
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        interpolation_group = QGroupBox("Interpolation Tools")
        tools_layout = QVBoxLayout()

        self.interpolate_next_segment_button = QPushButton("Generate Next Suggestion")
        self.interpolate_next_segment_button.setToolTip(
            "For the active person, find the next keyframe and interpolate the gap."
        )
        tools_layout.addWidget(self.interpolate_next_segment_button)

        self.auto_interpolate_checkbox = QCheckBox("Enable Auto-Interpolation")
        self.auto_interpolate_checkbox.setToolTip(
            "Automatically generate suggestions for all valid tracks when annotations change."
        )
        self.auto_interpolate_checkbox.setChecked(True) # Default to on
        tools_layout.addWidget(self.auto_interpolate_checkbox)

        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Max Gap (frames):"))
        self.interpolation_gap_spinbox = QSpinBox()
        self.interpolation_gap_spinbox.setRange(1, 1000)
        self.interpolation_gap_spinbox.setValue(30) # Default max gap
        gap_layout.addWidget(self.interpolation_gap_spinbox)
        tools_layout.addLayout(gap_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.interpolation_type_combobox = QComboBox()
        self.interpolation_type_combobox.addItems(["Linear"]) # "Cubic" can be added if implemented
        type_layout.addWidget(self.interpolation_type_combobox)
        tools_layout.addLayout(type_layout)

        self.accept_current_frame_interp_button = QPushButton("Accept Current Frame Interp.")
        self.accept_current_frame_interp_button.setToolTip(
            "Accept all purely interpolated (non-AI) suggestions on the current frame."
        )
        tools_layout.addWidget(self.accept_current_frame_interp_button)

        self.accept_all_frames_interp_button = QPushButton("Accept All Frames Interp.")
        self.accept_all_frames_interp_button.setToolTip(
            "Accept all purely interpolated (non-AI) suggestions on ALL frames."
        )
        tools_layout.addWidget(self.accept_all_frames_interp_button)

        self.clear_all_suggestions_button = QPushButton("Clear All Suggestions")
        self.clear_all_suggestions_button.setToolTip(
            "Removes all interpolated and AI-generated suggestions from all frames."
        )
        tools_layout.addWidget(self.clear_all_suggestions_button)

        interpolation_group.setLayout(tools_layout)
        main_layout.addWidget(interpolation_group)
        self.setLayout(main_layout)