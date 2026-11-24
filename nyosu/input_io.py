from __future__ import annotations

import ctypes
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import keyboard
except Exception:
    keyboard = None

from .pad_io import PadReader, PadWriter, pad_kind_of, pygame, standardize_raw, standardize_sdl, vg  # noqa: F401


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


_user32 = ctypes.windll.user32 if os.name == "nt" else None
_VK = {"lmb": 0x01, "rmb": 0x02}
_EXTRA = ctypes.c_ulong(0)
if _user32 is not None:
    try:
        _user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
        _user32.SendInput.restype = ctypes.c_uint
        _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        _user32.GetAsyncKeyState.restype = ctypes.c_short
    except Exception:
        pass


def _send_mouse(flags: int, dx: int = 0, dy: int = 0) -> None:
    if _user32 is None:
        return
    mi = _MOUSEINPUT(dx, dy, 0, flags, 0, ctypes.pointer(_EXTRA))
    inp = _INPUT(0, _INPUT._U(mi))
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def mouse_move(dx: float, dy: float) -> None:
    ix, iy = int(round(dx)), int(round(dy))
    if ix or iy:
        _send_mouse(0x0001, ix, iy)


def key_down(name: str) -> bool:
    vk = _VK.get(name.lower())
    if vk is None or _user32 is None:
        return False
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


