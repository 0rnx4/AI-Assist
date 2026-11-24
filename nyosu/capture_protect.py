from __future__ import annotations

import ctypes
import os

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _hwnd_of(widget) -> int:
    try:
        wid = widget.winId()
        return int(wid)
    except Exception:
        return 0


_affinity_fn = None


def _get_affinity_fn():
    global _affinity_fn
    if _affinity_fn is None:
        fn = ctypes.windll.user32.SetWindowDisplayAffinity
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        fn.restype = ctypes.c_int
        _affinity_fn = fn
    return _affinity_fn


def set_exclude_from_capture(widget, enabled: bool) -> bool:
    if os.name != "nt":
        return False
    hwnd = _hwnd_of(widget)
    if not hwnd:
        return False
    try:
        set_affinity = _get_affinity_fn()
    except Exception:
        return False
    if not enabled:
        try:
            return bool(set_affinity(ctypes.c_void_p(hwnd), WDA_NONE))
        except Exception:
            return False
    try:
        if set_affinity(ctypes.c_void_p(hwnd), WDA_EXCLUDEFROMCAPTURE):
            return True
    except Exception:
        pass
    try:
        return bool(set_affinity(ctypes.c_void_p(hwnd), WDA_MONITOR))
    except Exception:
        return False


def apply_capture_protection(widget, enabled: bool) -> bool:
    gen = getattr(widget, "_cap_gen", 0) + 1
    try:
        widget._cap_gen = gen
    except Exception:
        pass
    ok = set_exclude_from_capture(widget, enabled)
    if enabled and os.name == "nt":
        def _retry(gen=gen):
            if getattr(widget, "_cap_gen", None) == gen:
                set_exclude_from_capture(widget, True)

        try:
            from PyQt6.QtCore import QTimer
            for delay in (250, 1000):
                try:
                    QTimer.singleShot(delay, widget, _retry)
                except TypeError:
                    QTimer.singleShot(delay, _retry)
        except Exception:
            pass
    return ok
