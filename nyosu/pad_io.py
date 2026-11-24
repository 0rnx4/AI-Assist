from __future__ import annotations

import os
import threading

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from PyQt6.QtCore import QThread, pyqtSignal

from .config import log

try:
    import pygame
    pygame.init()
except Exception:
    pygame = None
try:
    import vgamepad as vg
except Exception:
    vg = None


def _clamp(v, lo=-1.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def _trig_from_axis(v):
    v = float(v)
    if v < 0.0:
        return (v + 1.0) / 2.0
    return v


def pad_kind_of(name: str) -> str:
    if any(k in name for k in ("DualSense", "DualShock", "Wireless Controller", "Sony", "PlayStation")):
        return "ps"
    return "xbox"


_XBOX_BTNS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 6: "back", 7: "start", 8: "l3", 9: "r3"}
_PS_BTNS = {0: "x", 1: "a", 2: "b", 3: "y", 4: "lb", 5: "rb", 6: "lt_btn", 7: "rt_btn", 8: "back", 9: "start", 10: "l3", 11: "r3"}


def standardize_raw(axes, buttons, hat, kind, trig_axes=None):
    def ax(i):
        return float(axes[i]) if i < len(axes) else 0.0

    def bt(i):
        return i < len(buttons) and bool(buttons[i])

    btns = set()
    table = _PS_BTNS if kind == "ps" else _XBOX_BTNS
    for i, name in table.items():
        if bt(i):
            btns.add(name)
    if hat is not None:
        hx, hy = hat
        if hy > 0:
            btns.add("dpad_up")
        elif hy < 0:
            btns.add("dpad_down")
        if hx < 0:
            btns.add("dpad_left")
        elif hx > 0:
            btns.add("dpad_right")
    if kind == "ps":
        lx, ly, rx, ry = ax(0), ax(1), ax(3), ax(4)
        lt = _trig_from_axis(ax(2)) if len(axes) > 2 else (1.0 if "lt_btn" in btns else 0.0)
        rt = _trig_from_axis(ax(5)) if len(axes) > 5 else (1.0 if "rt_btn" in btns else 0.0)
    else:
        lx, ly, rx, ry = ax(0), ax(1), ax(2), ax(3)
        if trig_axes:
            li, ri = trig_axes
            lt = _trig_from_axis(ax(li)) if li < len(axes) else 0.0
            rt = _trig_from_axis(ax(ri)) if ri < len(axes) else 0.0
        else:
            lt = 1.0 if "lt_btn" in btns else 0.0
            rt = 1.0 if "rt_btn" in btns else 0.0
    if lt > 0.5:
        btns.add("lt_btn")
    if rt > 0.5:
        btns.add("rt_btn")
    return {
        "lx": _clamp(lx), "ly": _clamp(-ly), "rx": _clamp(rx), "ry": _clamp(-ry),
        "lt": _clamp(lt, 0.0, 1.0), "rt": _clamp(rt, 0.0, 1.0),
        "buttons": frozenset(btns),
    }


def standardize_sdl(axes6, buttons15):
    def ax(i):
        return _clamp(float(axes6[i]) / 32767.0) if i < len(axes6) else 0.0

    names = ("a", "b", "x", "y", "back", "guide", "start", "l3", "r3", "lb", "rb",
             "dpad_up", "dpad_down", "dpad_left", "dpad_right")
    btns = {names[i] for i in range(min(len(buttons15), len(names))) if buttons15[i]}
    lt = _clamp(float(axes6[4]) / 32767.0, 0.0, 1.0) if len(axes6) > 4 else 0.0
    rt = _clamp(float(axes6[5]) / 32767.0, 0.0, 1.0) if len(axes6) > 5 else 0.0
    if lt > 0.5:
        btns.add("lt_btn")
    if rt > 0.5:
        btns.add("rt_btn")
    return {
        "lx": ax(0), "ly": -ax(1), "rx": ax(2), "ry": -ax(3),
        "lt": lt, "rt": rt, "buttons": frozenset(btns),
    }


_EMPTY_PAD = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0,
              "buttons": frozenset()}


class PadReader(QThread):
    state_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._evt = threading.Event()
        self._lock = threading.Lock()
        self._state = {"connected": False, "name": "", **_EMPTY_PAD}
        self._joy = None
        self._ctrl = None
        self._joy_name = ""
        self._joy_kind = "xbox"
        self._trigger_axes = None

    def stop(self):
        self._evt.set()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _detect_trigger_mapping(self, j):
        self._trigger_axes = None
        try:
            if j.get_numaxes() >= 6:
                a4, a5 = j.get_axis(4), j.get_axis(5)
                if (a4 < -0.5 and a5 < -0.5) or (0.0 <= a4 <= 1.0 and 0.0 <= a5 <= 1.0):
                    self._trigger_axes = (4, 5)
        except Exception:
            self._trigger_axes = None

    def _open_controller(self, idx):
        try:
            from pygame._sdl2 import controller as _ctl
            try:
                _ctl.init()
            except Exception:
                pass
            if not _ctl.is_controller(idx):
                return None
            c = _ctl.Controller(idx)
            name = ""
            try:
                name = c.name or ""
            except Exception:
                name = ""
            return c, name
        except Exception:
            return None

    def _connect(self):
        count = 0
        try:
            count = pygame.joystick.get_count()
        except Exception:
            count = 0
        for idx in range(count):
            got = self._open_controller(idx)
            if got is not None:
                self._ctrl, self._joy_name = got
                self._joy_kind = pad_kind_of(self._joy_name)
                with self._lock:
                    self._state = {"connected": True, "name": self._joy_name,
                                   "kind": self._joy_kind, **_EMPTY_PAD}
                self.state_changed.emit(self.snapshot())
                return True
        for idx in range(count):
            try:
                j = pygame.joystick.Joystick(idx)
                j.init()
            except Exception:
                continue
            self._joy = j
            self._detect_trigger_mapping(j)
            self._joy_name = j.get_name() or ""
            self._joy_kind = pad_kind_of(self._joy_name)
            with self._lock:
                self._state = {"connected": True, "name": self._joy_name,
                               "kind": self._joy_kind, **_EMPTY_PAD}
            self.state_changed.emit(self.snapshot())
            return True
        return False

    def _disconnect(self):
        if self._joy is not None:
            try:
                self._joy.quit()
            except Exception:
                pass
        if self._ctrl is not None:
            try:
                self._ctrl.close()
            except Exception:
                pass
        self._joy = None
        self._ctrl = None
        self._joy_name = ""
        self._trigger_axes = None
        with self._lock:
            self._state = {"connected": False, "name": "", **_EMPTY_PAD}
        self.state_changed.emit(self.snapshot())

    def _poll(self):
        if self._ctrl is not None:
            try:
                from pygame._sdl2 import controller as _ctl
                _ctl.update()
            except Exception:
                pass
            c = self._ctrl
            axes = [c.get_axis(i) for i in range(6)]
            btns = [c.get_button(i) for i in range(15)]
            return standardize_sdl(axes, btns)
        j = self._joy
        try:
            pygame.event.pump()
        except Exception:
            pass
        na, nb = j.get_numaxes(), j.get_numbuttons()
        axes = [j.get_axis(i) for i in range(na)]
        btns = [j.get_button(i) for i in range(nb)]
        hat = None
        try:
            if j.get_numhats() > 0:
                hat = j.get_hat(0)
        except Exception:
            hat = None
        return standardize_raw(axes, btns, hat, self._joy_kind, self._trigger_axes)

    def run(self):
        if pygame is None:
            self.state_changed.emit({"error": "pygameが利用できません"})
            return
        try:
            while not self._evt.is_set():
                if self._joy is None and self._ctrl is None:
                    if not self._connect():
                        self._evt.wait(0.5)
                        continue
                try:
                    st = self._poll()
                    st["connected"] = True
                    st["name"] = self._joy_name
                    st["kind"] = self._joy_kind
                    with self._lock:
                        self._state = st
                except Exception as e:
                    log.warning("パッド読み取りエラー: %s", e)
                    self._disconnect()
                self._evt.wait(0.008)
        finally:
            if self._joy is not None:
                try:
                    self._joy.quit()
                except Exception:
                    pass
            if self._ctrl is not None:
                try:
                    self._ctrl.close()
                except Exception:
                    pass
                self._ctrl = None


class PadWriter:
    _X360_MAP = None
    _DS4_MAP = None
    _DS4_DPAD = None

    def __init__(self, kind="x360"):
        self.kind = kind if kind in ("x360", "ds4") else "x360"
        self._pad = None
        self._warned = False
        self._last_sent = None
        self._sent_buttons = frozenset()
        if vg is not None:
            try:
                self._pad = vg.VDS4Gamepad() if self.kind == "ds4" else vg.VX360Gamepad()
            except Exception as e:
                log.info("vgamepad初期化失敗: %s", e)

    @property
    def available(self):
        return self._pad is not None

    @classmethod
    def _maps(cls):
        if cls._X360_MAP is None and vg is not None:
            X = vg.XUSB_BUTTON
            cls._X360_MAP = {
                "a": X.XUSB_GAMEPAD_A, "b": X.XUSB_GAMEPAD_B, "x": X.XUSB_GAMEPAD_X,
                "y": X.XUSB_GAMEPAD_Y, "lb": X.XUSB_GAMEPAD_LEFT_SHOULDER,
                "rb": X.XUSB_GAMEPAD_RIGHT_SHOULDER, "back": X.XUSB_GAMEPAD_BACK,
                "start": X.XUSB_GAMEPAD_START, "l3": X.XUSB_GAMEPAD_LEFT_THUMB,
                "r3": X.XUSB_GAMEPAD_RIGHT_THUMB, "dpad_up": X.XUSB_GAMEPAD_DPAD_UP,
                "dpad_down": X.XUSB_GAMEPAD_DPAD_DOWN, "dpad_left": X.XUSB_GAMEPAD_DPAD_LEFT,
                "dpad_right": X.XUSB_GAMEPAD_DPAD_RIGHT,
            }
            D = vg.DS4_BUTTONS
            cls._DS4_MAP = {
                "a": D.DS4_BUTTON_CROSS, "b": D.DS4_BUTTON_CIRCLE, "x": D.DS4_BUTTON_SQUARE,
                "y": D.DS4_BUTTON_TRIANGLE, "lb": D.DS4_BUTTON_SHOULDER_LEFT,
                "rb": D.DS4_BUTTON_SHOULDER_RIGHT, "back": D.DS4_BUTTON_SHARE,
                "start": D.DS4_BUTTON_OPTIONS, "l3": D.DS4_BUTTON_THUMB_LEFT,
                "r3": D.DS4_BUTTON_THUMB_RIGHT,
            }
            DD = vg.DS4_DPAD_DIRECTIONS
            cls._DS4_DPAD = {
                (0, 0): DD.DS4_BUTTON_DPAD_NONE, (0, 1): DD.DS4_BUTTON_DPAD_NORTH,
                (1, 1): DD.DS4_BUTTON_DPAD_NORTHEAST, (1, 0): DD.DS4_BUTTON_DPAD_EAST,
                (1, -1): DD.DS4_BUTTON_DPAD_SOUTHEAST, (0, -1): DD.DS4_BUTTON_DPAD_SOUTH,
                (-1, -1): DD.DS4_BUTTON_DPAD_SOUTHWEST, (-1, 0): DD.DS4_BUTTON_DPAD_WEST,
                (-1, 1): DD.DS4_BUTTON_DPAD_NORTHWEST,
            }
        return cls._X360_MAP, cls._DS4_MAP, cls._DS4_DPAD

    @staticmethod
    def _q(v):
        return int(_clamp(float(v)) * 32767)

    @staticmethod
    def _t(v):
        return int(_clamp(float(v), 0.0, 1.0) * 255)

    def full_update(self, base=None, assist_x=0.0, assist_y=0.0):
        if self._pad is None:
            return
        b = base or _EMPTY_PAD
        btns = frozenset(b.get("buttons") or ())
        state = (
            self._q(b.get("lx", 0.0)), self._q(b.get("ly", 0.0)),
            self._q(_clamp(float(b.get("rx", 0.0)) + float(assist_x))),
            self._q(_clamp(float(b.get("ry", 0.0)) + float(assist_y))),
            self._t(b.get("lt", 0.0)), self._t(b.get("rt", 0.0)),
            1 if "lt_btn" in btns else 0, 1 if "rt_btn" in btns else 0,
            btns,
        )
        if state == self._last_sent:
            return
        try:
            xmap, dmap, ddpad = self._maps()
            lx, ly, rx, ry, lt, rt, _ltb, _rtb, _btns = state
            self._pad.left_joystick(x_value=lx, y_value=ly)
            self._pad.right_joystick(x_value=rx, y_value=ry)
            self._pad.left_trigger(value=lt)
            self._pad.right_trigger(value=rt)
            new_sent = self._sent_buttons
            if self.kind == "ds4":
                eff = btns - {"dpad_up", "dpad_down", "dpad_left", "dpad_right", "lt_btn", "rt_btn", "guide"}
                if dmap:
                    for name in eff - self._sent_buttons:
                        if name in dmap:
                            self._pad.press_button(button=dmap[name])
                    for name in self._sent_buttons - eff:
                        if name in dmap:
                            self._pad.release_button(button=dmap[name])
                if ddpad:
                    dx = (1 if "dpad_right" in btns else 0) - (1 if "dpad_left" in btns else 0)
                    dy = (1 if "dpad_up" in btns else 0) - (1 if "dpad_down" in btns else 0)
                    self._pad.directional_pad(direction=ddpad[(dx, dy)])
                new_sent = eff
            else:
                eff = btns - {"lt_btn", "rt_btn", "guide"}
                if xmap:
                    for name in eff - self._sent_buttons:
                        if name in xmap:
                            self._pad.press_button(button=xmap[name])
                    for name in self._sent_buttons - eff:
                        if name in xmap:
                            self._pad.release_button(button=xmap[name])
                new_sent = eff
            self._pad.update()
            self._sent_buttons = new_sent
            self._last_sent = state
        except Exception as e:
            if not self._warned:
                self._warned = True
                log.warning("パッド出力に失敗しました: %s", e)

    def right_stick(self, x, y):
        self.full_update(None, float(x), float(y))

    def reset(self):
        if self._pad is None:
            return
        try:
            self._pad.reset()
            self._pad.update()
            self._warned = False
            self._last_sent = None
            self._sent_buttons = frozenset()
        except Exception as e:
            if not self._warned:
                self._warned = True
                log.warning("パッドのリセットに失敗しました: %s", e)
