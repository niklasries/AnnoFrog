# ==============================================================================
# UI PANEL WIDGETS
# ==============================================================================

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget, QVBoxLayout, QGroupBox, QComboBox, QPushButton, QDoubleSpinBox


class AIPosePanelWidget(QWidget):
    """
    Panel for AI Pose Estimation controls.
    Allows triggering AI inference and setting parameters.
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        ai_group = QGroupBox("AI Pose Estimation")
        ai_layout = QVBoxLayout()

        self.run_ai_button = QPushButton("Run AI on Current Frame (V)")
        self.run_ai_button.setToolTip("Run AI pose estimation on the currently displayed frame.")
        ai_layout.addWidget(self.run_ai_button)

        # Keypoint confidence threshold
        kp_conf_layout = QHBoxLayout()
        kp_conf_layout.addWidget(QLabel("Min KP Confidence:"))
        self.kp_confidence_spinbox = QDoubleSpinBox()
        self.kp_confidence_spinbox.setRange(0.0, 1.0)
        self.kp_confidence_spinbox.setSingleStep(0.05)
        self.kp_confidence_spinbox.setValue(0.3) # Default value from original Window class
        self.kp_confidence_spinbox.setToolTip("Minimum confidence score for an AI-detected keypoint to be used.")
        kp_conf_layout.addWidget(self.kp_confidence_spinbox)
        ai_layout.addLayout(kp_conf_layout)

        # AI Target Mode
        target_mode_layout = QHBoxLayout()
        target_mode_layout.addWidget(QLabel("Target Mode:"))
        self.target_mode_combobox = QComboBox()
        self.target_mode_combobox.addItems(["All Detected People", "Only for Empty User BBoxes"])
        self.target_mode_combobox.setToolTip(
            "'All Detected': Adds all new AI poses.\n"
            "'Only for Empty User BBoxes': Fills AI poses into user-drawn bboxes that have few keypoints."
        )
        target_mode_layout.addWidget(self.target_mode_combobox)
        ai_layout.addLayout(target_mode_layout)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        self.setLayout(layout)