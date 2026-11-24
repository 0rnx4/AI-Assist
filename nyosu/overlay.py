from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .capture_protect import apply_capture_protection


class Overlay(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.dets = []
        self.meta = {}
        self._capture_applied = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(QGuiApplication.primaryScreen().geometry())
        col = QColor(237, 237, 237, 175)
        self._pen_fov = QPen(col, 1.8)
        self._pen_box = QPen(col, 1.8)
        self._pen_line = QPen(QColor("#FF453A"), 1.8)
        self._pen_line_active = QPen(QColor("#FF453A"), 2.5)
        self._font = QFont("Consolas", 9)

    def refresh_capture_protection(self):
        want = bool(getattr(self.settings, "hide_overlay_from_capture", True))
        if self._capture_applied is not want:
            apply_capture_protection(self, want)
            self._capture_applied = want

    def showEvent(self, e):
        super().showEvent(e)
        self._capture_applied = None
        self.refresh_capture_protection()

    def update_data(self, dets, meta):
        if self.isVisible():
            self.refresh_capture_protection()
        clean = []
        for d in (dets or []):
            if not isinstance(d, dict) or "box" not in d or "conf" not in d:
                continue
            try:
                box = [float(v) for v in d["box"]]
                conf = float(d["conf"])
            except Exception:
                continue
            if len(box) == 4:
                clean.append({"box": box, "conf": conf})
        self.dets = clean
        self.meta = meta or {}
        self.update()

    def paintEvent(self, _e):
        if not self.settings.enabled:
            return
        dpr = self.devicePixelRatioF() or 1.0
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        if self.settings.show_fov:
            p.setPen(self._pen_fov)
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = max(self.settings.hip_fov, self.settings.ads_fov) / dpr
            p.drawEllipse(QPointF(cx, cy), r, r)

        if self.settings.show_boxes:
            p.setPen(self._pen_box)
            p.setFont(self._font)
            for d in self.dets:
                x1, y1, x2, y2 = [v / dpr for v in d["box"]]
                p.drawRoundedRect(QRectF(x1, y1, x2 - x1, y2 - y1), 6, 6)
                p.drawText(int(x1), max(12, int(y1) - 4), f'{d["conf"]:.2f}')

        if self.settings.show_target_line:
            t = self.meta.get("target")
            if t:
                p.setPen(self._pen_line_active if self.meta.get("aim_active") else self._pen_line)
                p.drawLine(QPointF(cx, cy), QPointF(t[0] / dpr, t[1] / dpr))
