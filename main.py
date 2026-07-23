"""Translation — Floating Translation Overlay."""

import sys
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from src.ui.floating_window import FloatingWindow


def _excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler — log and show message instead of crashing."""
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"ERROR: {tb_str}", file=sys.stderr)
    try:
        QMessageBox.critical(
            None, "错误",
            f"程序出错:\n\n{exc_value}\n\n详情已输出到终端。"
        )
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def main():
    # Catch all unhandled exceptions
    sys.excepthook = _excepthook

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Translation")

    window = FloatingWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
