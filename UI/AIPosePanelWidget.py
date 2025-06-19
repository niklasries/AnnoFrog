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

        self.run_ai_button = QPushButton("AI Detect (V)")
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
        detection_mode_layout = QHBoxLayout()
        detection_mode_layout.addWidget(QLabel("Detection Mode:"))
        self.detection_mode_combobox = QComboBox() 
        self.detection_mode_combobox.addItems([
            "RTMO - frame",          # Was "All Detected People"
            "RTMO - BBoxes", # Was "Only for Empty User BBoxes"
            "ViTPose - BBoxes"           # New ViTPose mode
        ])
        self.detection_mode_combobox.setToolTip(
            "'RTMO - All Detected': RTMO suggests poses for everyone it finds.\n"
            "'RTMO - Empty User BBoxes': RTMO fills poses into user-drawn bboxes with few keypoints.\n"
            "'ViTPose - User BBoxes': ViTPose estimates poses within all valid user-drawn bboxes on the frame."
        )
        detection_mode_layout.addWidget(self.detection_mode_combobox)
        ai_layout.addLayout(detection_mode_layout)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        self.setLayout(layout)

    
    def get_selected_detection_mode(self) -> str:
        return self.detection_mode_combobox.currentText()

    def get_kp_confidence(self) -> float:
        return self.kp_confidence_spinbox.value()