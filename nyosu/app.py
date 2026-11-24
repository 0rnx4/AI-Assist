from __future__ import annotations

import ctypes
import os
import sys
import threading

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from .config import log
from .main_window import MainWindow
from .theme import QSS


def _install_exception_hooks():
    def _hook(exc_type, exc, tb):
        log.error("未処理の例外", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):
        log.error("スレッド例外: %s", args.exc_value,
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _hook
    threading.excepthook = _thread_hook


def main():
    _install_exception_hooks()
    try:
        import faulthandler

        from .config import APP_DIR as _ad
        _cf = open(_ad / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(file=_cf, all_threads=True)
    except Exception:
        pass
    if os.name == "nt":
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
