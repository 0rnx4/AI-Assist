from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .tracking import target_point


class ToggleSwitch(QWidget):
    toggled = pyqtSignal(bool)

    _ON = QColor("#D84040")
    _OFF = QColor("#3A3A3A")
    _KNOB = QColor("#FFFFFF")

    def __init__(self, checked=False):
        super().__init__()
        self._checked = bool(checked)
        self.setFixedSize(52, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("クリックで ON / OFF")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        v = bool(v)
        if v != self._checked:
            self._checked = v
            self.update()
            self.toggled.emit(v)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.setChecked(not self._checked)
            e.accept()
            return
        super().keyPressEvent(e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.hasFocus():
            p.setPen(QPen(QColor("#D84040"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(0, 2, 51, 26, 13, 13)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._ON if self._checked else self._OFF)
        p.drawRoundedRect(1, 3, 50, 24, 12, 12)
        p.setBrush(self._KNOB)
        x = 26 if self._checked else 4
        p.drawEllipse(x, 5, 20, 20)


class OverlayPreview(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.setObjectName("OverlayPreview")
        self.setMinimumHeight(200)
        self._col = QColor(237, 237, 237, 170)
        self._pen_fov = QPen(self._col, 1.5)
        self._pen_line = QPen(self._col, 1.8)
        self._pen_red = QPen(QColor("#FF453A"), 1.8)
        self._font = QFont("Consolas", 9)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._timer.isActive():
            self._timer.start(500)

    def hideEvent(self, e):
        super().hideEvent(e)
        self._timer.stop()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        wpx, hpx = self.width(), self.height()
        cx, cy = wpx / 2.0, hpx / 2.0
        max_fov = max(self.settings.hip_fov, self.settings.ads_fov)
        r = max(20.0, min(wpx, hpx) * 0.5 * max_fov / 400.0)
        if self.settings.show_fov:
            p.setPen(self._pen_fov)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(self._pen_line)
        p.drawLine(QPointF(cx - 10, cy), QPointF(cx + 10, cy))
        p.drawLine(QPointF(cx, cy - 10), QPointF(cx, cy + 10))
        bx, by, bw, bh = cx + r * 0.45, cy - r * 0.55, 38.0, 56.0
        if self.settings.show_boxes:
            p.setPen(self._pen_line)
            p.setFont(self._font)
            p.drawRoundedRect(QRectF(bx, by, bw, bh), 6, 6)
            p.drawText(int(bx), max(12, int(by) - 4), "0.87")
        if self.settings.show_target_line:
            tx = bx + bw / 2
            if self.settings.aim_point_mode == "body":
                ty = by + bh * 0.38
            else:
                ty = target_point((bx, by, bx + bw, by + bh), "head")[1]
            p.setPen(self._pen_red)
            p.drawLine(QPointF(cx, cy), QPointF(tx, ty))


class PerfGraph(QWidget):
    def __init__(self, maxlen=180):
        super().__init__()
        self.setObjectName("PerfGraph")
        self.setMinimumHeight(86)
        self._maxlen = maxlen
        self._fps = []
        self._ms = []
        self._col_fps = QColor("#EDEDED")
        self._col_ms = QColor("#D84040")
        self._col_grid = QColor(255, 255, 255, 28)
        self._font = QFont("Consolas", 8)

    def push(self, fps, infer_ms):
        self._fps.append(float(fps))
        self._ms.append(float(infer_ms))
        if len(self._fps) > self._maxlen:
            del self._fps[0]
            del self._ms[0]
        if self.isVisible():
            self.update()

    def _line(self, p, vals, vmax, col):
        if len(vals) < 2:
            return
        wpx, hpx = self.width(), self.height()
        step = wpx / max(1, self._maxlen - 1)
        x0 = wpx - step * (len(vals) - 1)
        p.setPen(QPen(col, 1.6))
        prev = None
        for i, v in enumerate(vals):
            x = x0 + step * i
            y = hpx - 6 - max(0.0, min(1.0, v / vmax)) * (hpx - 22)
            pt = QPointF(x, y)
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        wpx, hpx = self.width(), self.height()
        p.setPen(QPen(self._col_grid, 1))
        for frac in (0.33, 0.66):
            y = hpx * frac
            p.drawLine(QPointF(0, y), QPointF(wpx, y))
        self._line(p, self._fps, 240.0, self._col_fps)
        self._line(p, self._ms, 50.0, self._col_ms)
        p.setFont(self._font)
        p.setPen(self._col_fps)
        p.drawText(8, 14, f"FPS {self._fps[-1]:.0f}" if self._fps else "FPS —")
        p.setPen(self._col_ms)
        p.drawText(76, 14, f"推論 {self._ms[-1]:.0f}ms" if self._ms else "推論 —")
