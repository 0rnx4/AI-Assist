from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout,
)

from .widgets import ToggleSwitch


class ControlsMixin:
    def _card(self, title, desc=""):
        f = QFrame()
        f.setObjectName("Card")
        l = QVBoxLayout(f)
        l.setContentsMargins(20, 18, 20, 18)
        l.setSpacing(11)
        t = QLabel(title)
        t.setObjectName("CardTitle")
        l.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("Muted")
            d.setWordWrap(True)
            l.addWidget(d)
        return f, l

    def _toggle(self, layout, label, val, cb, desc=""):
        row = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(1)
        lab = QLabel(label)
        lab.setObjectName("CtlLabel")
        texts.addWidget(lab)
        if desc:
            d = QLabel(desc)
            d.setObjectName("CtlDesc")
            d.setWordWrap(True)
            texts.addWidget(d)
        row.addLayout(texts, 1)
        sw = ToggleSwitch(val)
        sw.toggled.connect(cb)
        row.addWidget(sw)
        layout.addLayout(row)
        return sw

    def _slider(self, layout, label, lo, hi, val, suffix, cb, desc="", fmt=None):
        box = QVBoxLayout()
        box.setSpacing(2)
        top = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(1)
        lab = QLabel(label)
        lab.setObjectName("CtlLabel")
        texts.addWidget(lab)
        if desc:
            d = QLabel(desc)
            d.setObjectName("CtlDesc")
            d.setWordWrap(True)
            texts.addWidget(d)
        top.addLayout(texts, 1)
        fmtv = fmt or (lambda v: f"{v}{suffix}")
        badge = QLabel(fmtv(val))
        badge.setObjectName("Badge")
        badge.setCursor(Qt.CursorShape.PointingHandCursor)
        badge.setToolTip("クリックで数値を直接入力")
        top.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(int(val))
        s.valueChanged.connect(lambda v: (badge.setText(fmtv(v)), cb(v)))
        box.addWidget(s)
        layout.addLayout(box)

        def _badge_num(v):
            m = re.search(r"-?\d+(?:\.\d+)?", fmtv(v))
            return float(m.group()) if m else float(v)

        def _start_edit(_event):
            if top.indexOf(badge) < 0:
                return
            cur = s.value()
            try:
                scale = (_badge_num(cur) / cur) if cur else (_badge_num(hi) / hi if hi else 1.0)
            except Exception:
                scale = 1.0
            if not scale:
                scale = 1.0
            idx = top.indexOf(badge)
            badge.hide()
            top.removeWidget(badge)
            edit = QLineEdit(f"{_badge_num(cur):g}")
            edit.setObjectName("BadgeEdit")
            edit.setFixedWidth(64)
            edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            top.insertWidget(idx, edit)
            edit.setFocus()
            edit.selectAll()
            done = {"flag": False}

            def _finish(commit):
                if done["flag"]:
                    return
                done["flag"] = True
                if commit:
                    try:
                        typed = float(edit.text().strip())
                        native = int(round(typed / scale))
                        s.setValue(max(lo, min(hi, native)))
                    except Exception:
                        pass
                top.removeWidget(edit)
                edit.deleteLater()
                top.insertWidget(idx, badge)
                badge.setText(fmtv(s.value()))
                badge.show()

            edit.editingFinished.connect(lambda: _finish(True))

        badge.mousePressEvent = _start_edit
        return s

    def _combo(self, layout, label, items, current, cb):
        row = QHBoxLayout()
        l = QLabel(label)
        l.setObjectName("Muted")
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        c.currentTextChanged.connect(cb)
        row.addWidget(l)
        row.addWidget(c, 1)
        layout.addLayout(row)
        return c

    def _entry(self, layout, label, val, cb):
        row = QHBoxLayout()
        l = QLabel(label)
        l.setObjectName("Muted")
        e = QLineEdit(val)
        e.editingFinished.connect(lambda: cb(e.text()))
        row.addWidget(l)
        row.addWidget(e, 1)
        layout.addLayout(row)
        return e

    def _combo_map(self, layout, label, pairs, current, cb):
        row = QHBoxLayout()
        l = QLabel(label)
        l.setObjectName("Muted")
        c = QComboBox()
        for text, val in pairs:
            c.addItem(text, val)
        for i in range(c.count()):
            if c.itemData(i) == current:
                c.setCurrentIndex(i)
        c.currentIndexChanged.connect(lambda _: cb(c.currentData()))
        row.addWidget(l)
        row.addWidget(c, 1)
        layout.addLayout(row)
        return c

