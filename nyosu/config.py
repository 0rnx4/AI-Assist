from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace


APP_DIR = Path.home() / ".nyosu_ai"
try:
    APP_DIR.mkdir(exist_ok=True)
except Exception:
    import tempfile
    APP_DIR = Path(tempfile.gettempdir()) / ".nyosu_ai"
    try:
        APP_DIR.mkdir(exist_ok=True)
    except Exception:
        pass
SETTINGS_PATH = APP_DIR / "settings.json"
BACKUP_PATH = APP_DIR / "settings.json.bak"

log = logging.getLogger("nyosu")
log.setLevel(logging.INFO)
try:
    _fh = logging.FileHandler(APP_DIR / "nyosu.log", encoding="utf-8")
except Exception:
    _fh = logging.NullHandler()
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_fh)


def _ci(v, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return lo


def _cf(v, lo, hi):
    try:
        fv = float(v)
    except Exception:
        return lo
    if fv != fv:
        return lo
    return max(lo, min(hi, fv))


def _bb(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return default


def _ss(v, default=""):
    return v if isinstance(v, str) else default


_SETTINGS_FIELDS = None


def _known_fields():
    global _SETTINGS_FIELDS
    if _SETTINGS_FIELDS is None:
        _SETTINGS_FIELDS = frozenset(f.name for f in fields(Settings))
    return _SETTINGS_FIELDS


@dataclass
class Settings:
    enabled: bool = False
    model_path: str = ""
    class_ids: str = "0"

    crop_size: int = 640
    prediction: float = 1.0
    max_infer_fps: int = 0
    idle_infer_fps: int = 20
    overlay_fps: int = 30
    hold_ms: int = 120
    hip_enabled: bool = True
    hip_strength: float = 0.88
    hip_smoothing: int = 5
    hip_fov: int = 183
    hip_aim_height: float = 0.20
    hip_conf: float = 0.45
    ads_enabled: bool = True
    ads_strength: float = 0.65
    ads_smoothing: int = 6
    ads_fov: int = 151
    ads_aim_height: float = 0.16
    ads_conf: float = 0.60
    preset_name: str = "カスタム"
    auto_gain: bool = True
    monitor_index: int = 1
    sticky_ms: int = 80
    aim_curve: float = 1.0
    x_scale: float = 1.0
    y_scale: float = 1.0
    max_move_px: int = 0
    aim_delay_ms: int = 0
    jitter_px: float = 0.0
    recoil_x: float = 0.0
    recoil_y: float = 0.0
    recoil_enabled: bool = True
    deadzone: float = 1.0
    target_priority: str = "nearest"
    aim_point_mode: str = "head"


    aim_always_on: bool = False
    aim_keys: str = "rmb,lmb,lt,rt"
    activation_mode: str = "either"

    input_mode: str = "mouse"
    pad_sens: float = 0.85
    mouse_sens: float = 1.0

    show_fov: bool = True
    show_boxes: bool = True
    show_target_line: bool = True
    hide_overlay_from_capture: bool = True
    hide_gui_from_capture: bool = True
    always_on_top: bool = False

    pad_output: str = "x360"
    hotkey: str = "f2"
    panic_key: str = "f8"

    def validate(self) -> None:
        ci, cf, bb, ss = _ci, _cf, _bb, _ss
        self.crop_size = ci(self.crop_size, 320, 1280)
        self.prediction = cf(self.prediction, 0.0, 2.0)
        self.max_infer_fps = ci(self.max_infer_fps, 0, 240)
        self.idle_infer_fps = ci(self.idle_infer_fps, 1, 120)
        self.overlay_fps = ci(self.overlay_fps, 5, 120)
        self.hold_ms = ci(self.hold_ms, 0, 500)
        self.monitor_index = ci(self.monitor_index, 1, 8)
        self.hip_strength = cf(self.hip_strength, 0.05, 1.5)
        self.hip_smoothing = ci(self.hip_smoothing, 1, 20)
        self.hip_fov = ci(self.hip_fov, 40, 600)
        self.hip_aim_height = cf(self.hip_aim_height, 0.05, 0.45)
        self.hip_conf = cf(self.hip_conf, 0.05, 0.95)
        self.ads_strength = cf(self.ads_strength, 0.05, 1.5)
        self.ads_smoothing = ci(self.ads_smoothing, 1, 20)
        self.ads_fov = ci(self.ads_fov, 40, 600)
        self.ads_aim_height = cf(self.ads_aim_height, 0.05, 0.45)
        self.ads_conf = cf(self.ads_conf, 0.05, 0.95)
        self.auto_gain = bb(self.auto_gain, True)
        self.enabled = bb(self.enabled)
        self.hip_enabled = bb(self.hip_enabled, True)
        self.ads_enabled = bb(self.ads_enabled, True)
        self.aim_always_on = bb(self.aim_always_on)
        self.recoil_enabled = bb(self.recoil_enabled, True)
        self.show_fov = bb(self.show_fov, True)
        self.show_boxes = bb(self.show_boxes, True)
        self.show_target_line = bb(self.show_target_line, True)
        self.hide_overlay_from_capture = bb(self.hide_overlay_from_capture, True)
        self.hide_gui_from_capture = bb(self.hide_gui_from_capture, True)
        self.always_on_top = bb(self.always_on_top, False)
        self.model_path = ss(self.model_path)
        _raw_ids = ss(self.class_ids, "0")
        _ids = []
        for _part in _raw_ids.split(","):
            _part = _part.strip()
            if not _part:
                continue
            try:
                _v = int(_part)
            except Exception:
                continue
            if 0 <= _v <= 999 and _v not in _ids:
                _ids.append(_v)
        self.class_ids = ",".join(str(v) for v in _ids) if _ids else "0"
        _ak = []
        for _part in ss(self.aim_keys, "rmb,lmb,lt,rt").split(","):
            _part = _part.strip().lower()
            if _part in ("rmb", "lmb", "lt", "rt", "lb", "rb") and _part not in _ak:
                _ak.append(_part)
        self.aim_keys = ",".join(_ak) if _ak else "rmb,lmb,lt,rt"
        if self.activation_mode not in ("either", "both"):
            self.activation_mode = "either"
        self.hotkey = (ss(self.hotkey, "f2").strip().lower() or "f2")
        self.panic_key = (ss(self.panic_key, "f8").strip().lower() or "f8")
        if self.preset_name not in ("カスタム", "競技", "バランス", "軽量", "遠距離"):
            self.preset_name = "カスタム"
        self.sticky_ms = ci(self.sticky_ms, 0, 1000)
        self.recoil_x = cf(self.recoil_x, -200.0, 200.0)
        self.recoil_y = cf(self.recoil_y, 0.0, 400.0)
        self.deadzone = cf(self.deadzone, 0.0, 8.0)
        self.aim_curve = cf(self.aim_curve, 0.5, 2.0)
        self.x_scale = cf(self.x_scale, 0.3, 1.5)
        self.y_scale = cf(self.y_scale, 0.3, 1.5)
        self.max_move_px = ci(self.max_move_px, 0, 200)
        self.aim_delay_ms = ci(self.aim_delay_ms, 0, 300)
        self.jitter_px = cf(self.jitter_px, 0.0, 5.0)
        if self.target_priority not in ("nearest", "biggest", "conf"):
            self.target_priority = "nearest"
        if self.aim_point_mode not in ("head", "body", "random"):
            self.aim_point_mode = "head"
        self.pad_sens = cf(self.pad_sens, 0.1, 3.0)
        self.mouse_sens = cf(self.mouse_sens, 0.05, 5.0)
        if self.input_mode not in ("mouse", "pad", "converter"):
            self.input_mode = "mouse"
        if self.pad_output not in ("x360", "ds4"):
            self.pad_output = "x360"
        self.ads_enabled = self.hip_enabled
        self.ads_strength = self.hip_strength
        self.ads_smoothing = self.hip_smoothing
        self.ads_fov = self.hip_fov
        self.ads_aim_height = self.hip_aim_height
        self.ads_conf = self.hip_conf

    def snapshot(self):
        return copy.copy(self)

    def aim_profile(self, mode):
        if mode == "ads":
            return SimpleNamespace(
                mode="ads", enabled=self.hip_enabled, strength=self.hip_strength,
                smoothing=self.hip_smoothing, fov=self.hip_fov,
                aim_height=self.hip_aim_height, conf=self.hip_conf)
        return SimpleNamespace(
            mode="hip", enabled=self.hip_enabled, strength=self.hip_strength,
            smoothing=self.hip_smoothing, fov=self.hip_fov,
            aim_height=self.hip_aim_height, conf=self.hip_conf)

    def base_confidence(self):
        return min(self.hip_conf, self.ads_conf)


def _autodetect_model(s):
    if s.model_path and Path(s.model_path).exists():
        return s
    root = Path(__file__).resolve().parent.parent
    for name in ("apex_v8s_3class.onnx", "Apex_Legends_by_5pr_2_c2g6.onnx"):
        cand = root / name
        if cand.exists():
            s.model_path = str(cand)
            break
    return s


def _read_settings_dict(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def load_settings() -> Settings:
    s = Settings()
    if not SETTINGS_PATH.exists():
        return _autodetect_model(s)
    data = None
    try:
        data = _read_settings_dict(SETTINGS_PATH)
    except Exception as e:
        log.warning("設定ファイルの読み込みに失敗しました: %s", e)
    if data is None:
        try:
            data = _read_settings_dict(BACKUP_PATH)
            if data is not None:
                log.warning("設定ファイルが壊れていたため、バックアップから復元しました")
        except Exception:
            data = None
    if data is None:
        log.warning("設定ファイルとバックアップが使えないため既定値を使います")
        return _autodetect_model(s)
    known = _known_fields()
    for k, v in data.items():
        if k in known:
            setattr(s, k, v)
    s.validate()
    return _autodetect_model(s)


def save_settings(s: Settings) -> bool:
    s.validate()
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    try:
        if SETTINGS_PATH.exists():
            try:
                if _read_settings_dict(SETTINGS_PATH) is not None:
                    _btmp = BACKUP_PATH.with_suffix(".tmp")
                    _btmp.write_bytes(SETTINGS_PATH.read_bytes())
                    os.replace(_btmp, BACKUP_PATH)
            except Exception:
                pass
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s), indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SETTINGS_PATH)
        return True
    except Exception as e:
        log.warning("設定の保存に失敗: %s", e)
        return False
