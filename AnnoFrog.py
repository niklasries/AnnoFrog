import sys


from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurfaceFormat, QIcon
from PyQt6.QtWidgets import QApplication

from UI.Window import Window

from pathlib import Path
import os
# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parent = Path(__file__).parent
    # Request specific OpenGL format
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL) # Ensures desktop GL is used
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3) # OpenGL 3.3
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile) # Core profile
    gl_format.setSamples(4)      # Enable 4x MSAA for antialiasing
    gl_format.setSwapInterval(1) # Enable VSync (usually 1 for on, 0 for off)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(parent)+'/res/UI/AnnoFrog.png'))
    main_window = Window()
    main_window.show()
    sys.exit(app.exec())