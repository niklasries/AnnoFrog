from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap,QMouseEvent

from PyQt6.QtWidgets import QLabel, QWidget

class ClickableLabel(QLabel):
    """A QLabel that emits a clicked signal with its frame_id when clicked."""
    clicked = pyqtSignal(int)  # Signal emitted: frame_id

    def __init__(self, frame_id: int, pixmap: QPixmap, parent: QWidget | None = None):
        """
        Initializes the clickable label.
        Args:
            frame_id: The frame ID this label represents.
            pixmap: The QPixmap to display on the label.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.frame_id = frame_id
        self.setPixmap(pixmap)
        self.setToolTip(f"Frame {frame_id + 1}") # Display frame number (1-indexed)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse press events to emit the clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.frame_id)
        super().mousePressEvent(event)