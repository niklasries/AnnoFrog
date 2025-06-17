# ==============================================================================
# UI PANEL WIDGETS
# ==============================================================================

from PyQt6.QtCore import Qt, pyqtSignal

from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QGroupBox, QMessageBox, QPushButton, QLineEdit


class DeleteAnnotationsPanel(QWidget):
    """Panel for deleting annotations by Person IDs and frame ranges."""
    delete_requested = pyqtSignal(list, list) # Emits (target_ids_int_list, frame_specs_list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        group = QGroupBox("Delete Annotations")
        group_layout = QVBoxLayout()

        group_layout.addWidget(QLabel("Person IDs (comma-separated):"))
        self.ids_input = QLineEdit()
        self.ids_input.setPlaceholderText("e.g., 1, 3, 5")
        group_layout.addWidget(self.ids_input)

        group_layout.addWidget(QLabel("Frames (e.g., 1-10, 15, 22-30 or 'all'):"))
        self.frames_input = QLineEdit()
        self.frames_input.setPlaceholderText("1-10, 15, all (1-indexed)")
        group_layout.addWidget(self.frames_input)

        self.delete_button = QPushButton("Delete Specified Annotations")
        self.delete_button.setToolTip("Deletes annotations matching the criteria. This action is irreversible.")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        group_layout.addWidget(self.delete_button)

        group.setLayout(group_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    def _on_delete_clicked(self):
        """Parses input fields and emits the delete_requested signal if valid."""
        ids_str = self.ids_input.text()
        frames_str = self.frames_input.text()

        try:
            target_ids = [int(id_val.strip()) for id_val in ids_str.split(',') if id_val.strip().isdigit()]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid Person IDs. Please use comma-separated numbers.")
            return

        frame_specs = [] # List of tuples (start_frame_0_idx, end_frame_0_idx) or "all"
        if not frames_str.strip() or frames_str.strip().lower() == "all":
            frame_specs.append("all")
        else:
            parts = frames_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part: # Range
                    s_str, e_str = part.split('-', 1)
                    if s_str.isdigit() and e_str.isdigit():
                        # Convert 1-indexed from UI to 0-indexed for internal use
                        frame_specs.append((int(s_str) - 1, int(e_str) - 1))
                    else:
                        QMessageBox.warning(self, "Input Error", f"Invalid frame range: {part}. Use numbers like '10-20'.")
                        return
                elif part.isdigit(): # Single frame
                    frame_specs.append((int(part) - 1, int(part) - 1)) # 0-indexed
                else:
                    QMessageBox.warning(self, "Input Error", f"Invalid frame specification: {part}. Use numbers or ranges.")
                    return

        if not target_ids:
            QMessageBox.warning(self, "Input Error", "No Person IDs specified for deletion.")
            return
        if not frame_specs: # Should be caught by "all" or specific parsing
            QMessageBox.warning(self, "Input Error", "No frames specified for deletion.")
            return

        self.delete_requested.emit(target_ids, frame_specs)