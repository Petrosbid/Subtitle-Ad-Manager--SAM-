import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont , QIcon
from gui import AdManagerWindow
import ctypes

def main():
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mycompany.myproduct.subadmanager.v1.0")
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setWindowIcon(QIcon("images/logo.ico"))
    window = AdManagerWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()