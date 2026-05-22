import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from gui import AdManagerWindow

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = AdManagerWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()