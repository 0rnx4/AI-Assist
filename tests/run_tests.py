from __future__ import annotations

import os
import sys
import tempfile
import time
import types

_TMP = tempfile.mkdtemp(prefix="nyosu_test_")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_fake_mss = types.ModuleType("mss")


class _Shot:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.bgra = bytes(w * h * 4)

    def __array__(self, dtype=None):
        return np.frombuffer(self.bgra, dtype=np.uint8).reshape(self.height, self.width, 4)


class _Ctx:
    monitors = [{"left": 0, "top": 0, "width": 1024, "height": 768},
                {"left": 0, "top": 0, "width": 1024, "height": 768}]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self, mon):
        return _Shot(int(mon["width"]), int(mon["height"]))


_fake_mss.mss = lambda: _Ctx()
sys.modules.setdefault("mss", _fake_mss)

RESULTS = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL {name}: {type(e).__name__}: {e}")


def t_settings_junk():
    from nyosu.config import Settings
    junk = [None, True, 0, -1, 10 ** 30, 1.5, float("inf"), float("nan"),
            "", "abc", "0,1", [], [1], {}, {"a": 1}, "x" * 5000]
    names = list(Settings().__dict__)
    for i, name in enumerate(names):
        s = Settings()
        setattr(s, name, junk[i % len(junk)])
        s.validate()
        s.aim_profile("hip")
        s.aim_profile("ads")
        s.base_confidence()


def t_validate_boundaries():
    from nyosu.config import Settings
    s = Settings()
    s.crop_size = 319
    s.validate()
    assert s.crop_size == 320
    s.crop_size = 1281
    s.validate()
    assert s.crop_size == 1280
    s.hip_conf = 0.0
    s.ads_conf = 1.0
    s.validate()
    assert s.hip_conf == 0.05 and s.ads_conf == s.hip_conf
    s.hip_smoothing = 0
    s.prediction = 99.0
    s.deadzone = -3.0
    s.validate()
    assert s.hip_smoothing == 1 and s.prediction == 2.0 and s.deadzone == 0.0
    s.monitor_index = 0
    s.max_infer_fps = -5
    s.validate()
    assert s.monitor_index == 1 and s.max_infer_fps == 0
    s.recoil_enabled = "no"
    s.validate()
    assert s.recoil_enabled is False
    s.recoil_enabled = None
    s.validate()
    assert s.recoil_enabled is True


def t_class_ids_sanitize():
    from nyosu.config import Settings
    s = Settings()
    s.class_ids = "0, abc, 2, 2, -1, 1000, 1"
    s.validate()
    assert s.class_ids == "0,2,1", s.class_ids
    s.class_ids = ""
    s.validate()
    assert s.class_ids == "0"
    s.class_ids = "xyz"
    s.validate()
    assert s.class_ids == "0"
    s.class_ids = " 3 , 0 "
    s.validate()
    assert s.class_ids == "3,0", s.class_ids


def t_snapshot_isolation():
    from nyosu.config import Settings
    s = Settings()
    s.hip_fov = 200
    snap = s.snapshot()
    s.hip_fov = 90
    assert snap.hip_fov == 200
    snap.hip_fov = 55
    assert s.hip_fov == 90


def t_corrupt_settings_file():
    import json as _json
    from nyosu.config import SETTINGS_PATH, load_settings
    try:
        SETTINGS_PATH.write_text("{broken json", encoding="utf-8")
        s = load_settings()
        assert s.hip_strength == 0.88
        SETTINGS_PATH.write_text("[1,2,3]", encoding="utf-8")
        s = load_settings()
        assert s.hip_strength == 0.88
        SETTINGS_PATH.write_text(_json.dumps({"hip_strength": "bad", "crop_size": None, "enabled": "yes"}), encoding="utf-8")
        s = load_settings()
        assert s.hip_strength == 0.05 and s.crop_size == 320 and s.enabled is True
    finally:
        try:
            SETTINGS_PATH.unlink()
        except Exception:
            pass


def t_detector_bad_models():
    import tempfile as _tf
    from nyosu.config import Settings
    from nyosu.vision import Detector
    s = Settings()
    s.model_path = "/nonexistent/none.onnx"
    d = Detector()
    d.load(s)
    assert d.session is None and d.error
    with _tf.NamedTemporaryFile("wb", suffix=".onnx", delete=False) as f:
        f.write(b"this is not an onnx file")
        bad = f.name
    s.model_path = bad
    d.load(s)
    assert d.session is None and d.error
    s.model_path = bad + ".txt"
    d.load(s)
    assert d.session is None and d.error


def t_settings_roundtrip():
    from nyosu.config import Settings, load_settings, save_settings
    s = Settings()
    s.hip_strength = 0.77
    s.model_path = "/nonexistent/deleted.onnx"
    save_settings(s)
    s2 = load_settings()
    assert abs(s2.hip_strength - 0.77) < 1e-9
    assert s2.model_path.endswith("apex_v8s_3class.onnx"), s2.model_path


def t_postprocess():
    from nyosu.config import Settings
    from nyosu.vision import Detector
    s = Settings()
    d = Detector()
    _rng = np.random.default_rng(1234)
    for shp in [(1, 7, 8400), (1, 5, 8400), (7, 8400), (1, 4, 100), (1, 6, 3)]:
        pred = _rng.random(shp).astype(np.float32)
        if shp[-2] > 4:
            pred[..., :4, :] *= 640.0
            pred.flat[::7] = np.nan
        out = d._postprocess(pred, 1.0, 0, 0, 640, 640, s)
        for det in out:
            x1, y1, x2, y2 = det["box"]
            assert 0 <= x1 <= 639 and 0 <= y1 <= 639
            assert 0.0 <= det["conf"] <= 1.0


def t_kalman():
    from nyosu.tracking import KalmanCV
    k = KalmanCV()
    t = 0.0
    for i in range(240):
        t = i / 120.0
        k.update((100.0 + 600.0 * t, 200.0), t)
    assert abs(k.vx - 600.0) < 60.0, k.vx
    assert abs(k.vy) < 30.0, k.vy


def t_det_tracker():
    from nyosu.tracking import DetTracker
    tr = DetTracker()
    tr.update([{"box": [100, 100, 160, 220], "conf": 0.9}], 0.0)
    out = tr.update([{"box": [110, 100, 170, 220], "conf": 0.9}], 0.05)
    assert len(out) == 1
    assert 100 < out[0]["box"][0] < 110, out[0]["box"]


def t_detector_models():
    from nyosu.config import Settings
    from nyosu.vision import Detector
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = np.zeros((480, 480, 3), dtype=np.uint8)
    for name in ("apex_v8s_3class.onnx", "apex_v8m_accurate.onnx", "Apex_Legends_by_5pr_2_c2g6.onnx"):
        s = Settings()
        s.model_path = os.path.join(root, name)
        d = Detector()
        d.load(s)
        assert d.error is None and d.session is not None, (name, d.error)
        assert d.input_size == 640
        dets = d.infer(img, s, 480, 480)
        assert isinstance(dets, list)


def _make_window():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from nyosu.main_window import MainWindow
    return app, MainWindow()


def t_config_guard_and_test_detection():
    import json as _json
    app, w = _make_window()
    assert w._apply_config_dict([1, 2, 3]) is False
    assert w._apply_config_dict("text") is False
    assert w._apply_config_dict({"hip_strength": 0.5}) is True
    cfg_dir = w._config_dir()
    (cfg_dir / "badcfg.json").write_text("[1,2,3]", encoding="utf-8")
    (cfg_dir / "goodcfg.json").write_text(_json.dumps({"hip_strength": 0.61}), encoding="utf-8")
    w._refresh_named_configs()
    w._named_combo.setCurrentText("badcfg")
    w._load_named_config()
    txt = w.log_view.toPlainText()
    assert "名前付き設定を読み込みました: badcfg" not in txt
    w._named_combo.setCurrentText("goodcfg")
    w._load_named_config()
    txt = w.log_view.toPlainText()
    assert "名前付き設定を読み込みました: goodcfg" in txt
    assert abs(w.settings.hip_strength - 0.61) < 1e-9
    w._test_detection()
    txt = w.log_view.toPlainText()
    assert "形式が不正" in txt
    assert any("テスト検出:" in line or "テスト検出: " in line for line in txt.splitlines()), txt[-300:]
    assert "has no attribute 'provider'" not in txt
    w.close()
    app.processEvents()


def _make_engine(poison=False):
    from PyQt6.QtCore import Qt
    from nyosu.config import Settings
    from nyosu.engine import AimEngine
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = Settings()
    s.enabled = True
    s.model_path = os.path.join(root, "apex_v8s_3class.onnx")
    e = AimEngine(s, None)
    logs = []
    e.log_sig.connect(lambda m: logs.append(m), Qt.ConnectionType.DirectConnection)
    if poison:
        def _boom(frame, st, ow, oh):
            raise RuntimeError("test-boom")
        e.detector.infer = _boom
    return e, logs


def t_engine_cycle():
    e, _logs = _make_engine()
    e.start()
    time.sleep(1.5)
    e.stop()
    assert e.wait(10000)
    e.start()
    time.sleep(1.5)
    e.stop()
    assert e.wait(10000)


def t_engine_worker_death():
    e, logs = _make_engine(poison=True)
    e.start()
    assert e.wait(30000), "engine did not exit after worker death"
    assert any("推論ワーカーが停止" in m for m in logs), logs


def t_engine_transient_failures():
    e, _logs = _make_engine()
    orig = e.detector.infer
    state = {"n": 0}

    def flaky(frame, st, ow, oh):
        state["n"] += 1
        if state["n"] <= 5:
            raise RuntimeError("flaky-test")
        return orig(frame, st, ow, oh)

    e.detector.infer = flaky
    e.start()
    time.sleep(6.0)
    alive = e.isRunning()
    e.stop()
    assert e.wait(10000)
    assert alive, "engine died on transient inference failures"
    assert state["n"] > 5


def t_engine_long_run_race():
    import threading as _th
    e, _logs = _make_engine()
    stop_flag = {"stop": False}

    def mutator():
        vals = [120, 200, 183, 90, 300]
        i = 0
        while not stop_flag["stop"]:
            e.settings.hip_fov = vals[i % len(vals)]
            e.settings.ads_fov = vals[(i + 2) % len(vals)]
            e.settings.prediction = 0.5 + (i % 10) * 0.1
            e.settings.target_priority = ["nearest", "biggest", "conf"][i % 3]
            e.settings.class_ids = ["0", "0,1", "0,1,2"][i % 3]
            i += 1
            time.sleep(0.01)

    e.start()
    th = _th.Thread(target=mutator, daemon=True)
    th.start()
    time.sleep(20.0)
    alive = e.isRunning()
    stop_flag["stop"] = True
    th.join(3.0)
    e.stop()
    assert e.wait(15000)
    assert alive, "engine died during long run with concurrent settings writes"

def t_capture_settings():
    from nyosu.config import Settings
    s = Settings()
    assert s.hide_overlay_from_capture is True and s.hide_gui_from_capture is True
    s.hide_overlay_from_capture = "no"
    s.hide_gui_from_capture = None
    s.validate()
    assert s.hide_overlay_from_capture is False
    assert s.hide_gui_from_capture is True


def t_capture_protect_noop():
    import os as _os
    from nyosu.capture_protect import WDA_EXCLUDEFROMCAPTURE, apply_capture_protection, set_exclude_from_capture
    assert WDA_EXCLUDEFROMCAPTURE == 0x11
    if _os.name != "nt":
        from PyQt6.QtWidgets import QApplication, QWidget
        app = QApplication.instance() or QApplication(sys.argv)
        w = QWidget()
        w.show()
        app.processEvents()
        assert set_exclude_from_capture(w, True) is False
        assert apply_capture_protection(w, True) is False
        w.close()


def t_nms_equivalence():
    from nyosu.vision import nms, nms_fast
    rng = np.random.default_rng(42)
    for _ in range(30):
        n = int(rng.integers(1, 200))
        xyxy = rng.random((n, 4), dtype=np.float32) * 500
        xyxy[:, 2] = xyxy[:, 0] + rng.random(n, dtype=np.float32) * 120
        xyxy[:, 3] = xyxy[:, 1] + rng.random(n, dtype=np.float32) * 120
        scores = rng.random(n, dtype=np.float32)
        assert sorted(nms(xyxy, scores, 0.45).tolist()) == sorted(nms_fast(xyxy, scores, 0.45).tolist())


def t_tracker_malformed_and_ema():
    from nyosu.tracking import DetTracker
    tr = DetTracker()
    tr.update([{"box": [100, 100, 160, 220], "conf": 0.9}], 0.0)
    out = tr.update([{"box": [110, 100, 170, 220], "conf": 0.9}], 0.05)
    exp = [int(round(0.6 * 110 + 0.4 * 100)), 100, int(round(0.6 * 170 + 0.4 * 160)), 220]
    assert out[0]["box"] == exp, (out, exp)
    out = tr.update([{"box": [1, 2], "conf": 0.5}, {"nope": 1}, {"box": [120, 100, 180, 220], "conf": 0.8}], 0.1)
    assert len(out) == 1


def t_validate_nan_inf():
    from nyosu.config import Settings
    s = Settings()
    s.hip_strength = float("nan")
    s.ads_strength = float("inf")
    s.prediction = float("nan")
    s.validate()
    assert s.hip_strength == 0.05 and s.ads_strength == s.hip_strength and s.prediction == 0.0


def t_postprocess_guards():
    from nyosu.config import Settings
    from nyosu.vision import Detector
    s = Settings()
    d = Detector()
    assert d._postprocess(np.zeros((5,), dtype=np.float32), 1.0, 0, 0, 640, 640, s) == []
    assert d._postprocess(np.zeros((2, 3, 4, 5), dtype=np.float32), 1.0, 0, 0, 640, 640, s) == []
    assert d._postprocess(np.zeros((1, 7, 0), dtype=np.float32), 1.0, 0, 0, 640, 640, s) == []


def t_padwriter_dedupe():
    import nyosu.input_io as io

    class _FakePad:
        def __init__(self):
            self.updates = 0
            self.calls = []

        def left_joystick(self, x_value, y_value):
            self.calls.append(("lj", x_value, y_value))

        def right_joystick(self, x_value, y_value):
            self.calls.append(("rj", x_value, y_value))

        def left_trigger(self, value):
            self.calls.append(("lt", value))

        def right_trigger(self, value):
            self.calls.append(("rt", value))

        def press_button(self, button):
            self.calls.append(("press", button))

        def release_button(self, button):
            self.calls.append(("rel", button))

        def directional_pad(self, direction):
            self.calls.append(("dpad", direction))

        def update(self):
            self.updates += 1

        def reset(self):
            pass

    pw = io.PadWriter.__new__(io.PadWriter)
    pw.kind = "x360"
    pw._pad = _FakePad()
    pw._warned = False
    pw._last_sent = None
    pw._sent_buttons = frozenset()
    pw.right_stick(0.5, 0.5)
    pw.right_stick(0.5, 0.5)
    pw.right_stick(0.5, 0.5)
    assert pw._pad.updates == 1, pw._pad.updates
    pw.right_stick(0.25, 0.5)
    assert pw._pad.updates == 2
    pw.reset()
    assert pw._last_sent is None
    base = pw._pad.updates
    pw.right_stick(0.25, 0.5)
    assert pw._pad.updates == base + 1


def t_model_info_cache():
    from nyosu.main_window import _MODEL_INFO_CACHE
    app, w = _make_window()
    w._refresh_model_info()
    txt1 = w.model_info.text()
    n1 = len(_MODEL_INFO_CACHE)
    assert "入力" in txt1 and n1 >= 1, txt1
    w._refresh_model_info()
    assert w.model_info.text() == txt1
    assert len(_MODEL_INFO_CACHE) == n1
    w.close()
    app.processEvents()


def t_pad_standardize():
    from nyosu.input_io import standardize_raw, standardize_sdl
    st = standardize_raw([0.5, -0.25, 0.1, 0.2, -1.0, -1.0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], (0, 1), "xbox", (4, 5))
    assert abs(st["lx"] - 0.5) < 1e-9 and abs(st["ly"] - 0.25) < 1e-9
    assert abs(st["ry"] + 0.2) < 1e-9 and st["lt"] == 0.0
    assert {"a", "rb", "dpad_up"} <= st["buttons"]
    st = standardize_raw([0.0, 0.0, 1.0, 0.3, 0.4, 1.0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (-1, 0), "ps", None)
    assert st["lt"] == 1.0 and st["rt"] == 1.0 and abs(st["rx"] - 0.3) < 1e-9
    assert {"a", "dpad_left", "lt_btn", "rt_btn"} <= st["buttons"]
    st = standardize_sdl([16384, -16384, 0, 0, 32767, 0], [1] + [0] * 14)
    assert abs(st["lx"] - 0.5) < 0.01 and abs(st["ly"] - 0.5) < 0.01 and st["lt"] > 0.99
    assert "a" in st["buttons"] and "lt_btn" in st["buttons"]


def t_padwriter_full_forward():
    import types as _types

    import nyosu.input_io as io

    class _E:
        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return hash(self.v)

        def __eq__(self, o):
            return getattr(o, "v", None) == self.v

    class _FakePad:
        def __init__(self):
            self.calls = []
            self.updates = 0

        def left_joystick(self, x_value, y_value):
            self.calls.append(("lj", x_value, y_value))

        def right_joystick(self, x_value, y_value):
            self.calls.append(("rj", x_value, y_value))

        def left_trigger(self, value):
            self.calls.append(("lt", value))

        def right_trigger(self, value):
            self.calls.append(("rt", value))

        def press_button(self, button):
            self.calls.append(("press", button.v))

        def release_button(self, button):
            self.calls.append(("rel", button.v))

        def directional_pad(self, direction):
            self.calls.append(("dpad", direction.v))

        def update(self):
            self.updates += 1

        def reset(self):
            pass

    names_x = ["XUSB_GAMEPAD_A", "XUSB_GAMEPAD_B", "XUSB_GAMEPAD_X", "XUSB_GAMEPAD_Y",
               "XUSB_GAMEPAD_LEFT_SHOULDER", "XUSB_GAMEPAD_RIGHT_SHOULDER", "XUSB_GAMEPAD_BACK",
               "XUSB_GAMEPAD_START", "XUSB_GAMEPAD_LEFT_THUMB", "XUSB_GAMEPAD_RIGHT_THUMB",
               "XUSB_GAMEPAD_DPAD_UP", "XUSB_GAMEPAD_DPAD_DOWN", "XUSB_GAMEPAD_DPAD_LEFT", "XUSB_GAMEPAD_DPAD_RIGHT"]
    names_d = ["DS4_BUTTON_CROSS", "DS4_BUTTON_CIRCLE", "DS4_BUTTON_SQUARE", "DS4_BUTTON_TRIANGLE",
               "DS4_BUTTON_SHOULDER_LEFT", "DS4_BUTTON_SHOULDER_RIGHT", "DS4_BUTTON_SHARE",
               "DS4_BUTTON_OPTIONS", "DS4_BUTTON_THUMB_LEFT", "DS4_BUTTON_THUMB_RIGHT"]
    names_dd = ["DS4_BUTTON_DPAD_NONE", "DS4_BUTTON_DPAD_NORTH", "DS4_BUTTON_DPAD_NORTHEAST",
                "DS4_BUTTON_DPAD_EAST", "DS4_BUTTON_DPAD_SOUTHEAST", "DS4_BUTTON_DPAD_SOUTH",
                "DS4_BUTTON_DPAD_SOUTHWEST", "DS4_BUTTON_DPAD_WEST", "DS4_BUTTON_DPAD_NORTHWEST"]
    import nyosu.pad_io as pio
    old_vg = io.vg
    old_pio_vg = pio.vg
    io.vg = _types.SimpleNamespace(
        XUSB_BUTTON=_types.SimpleNamespace(**{k: _E(k) for k in names_x}),
        DS4_BUTTONS=_types.SimpleNamespace(**{k: _E(k) for k in names_d}),
        DS4_DPAD_DIRECTIONS=_types.SimpleNamespace(**{k: _E(k) for k in names_dd}))
    pio.vg = io.vg
    io.PadWriter._X360_MAP = io.PadWriter._DS4_MAP = io.PadWriter._DS4_DPAD = None
    try:
        base = {"lx": 0.5, "ly": 0.0, "rx": 0.25, "ry": 0.0, "lt": 1.0, "rt": 0.0,
                "buttons": frozenset({"a", "dpad_up"})}
        for kind, press_name, dpad_name in [("x360", "XUSB_GAMEPAD_A", "XUSB_GAMEPAD_DPAD_UP"),
                                            ("ds4", "DS4_BUTTON_CROSS", "DS4_BUTTON_DPAD_NORTH")]:
            pw = io.PadWriter.__new__(io.PadWriter)
            pw.kind = kind
            pw._pad = _FakePad()
            pw._warned = False
            pw._last_sent = None
            pw._sent_buttons = frozenset()
            pw.full_update(base, 0.25, 0.0)
            c = pw._pad.calls
            assert ("lj", int(0.5 * 32767), 0) in c, c
            assert ("rj", int(0.5 * 32767), 0) in c, c
            assert ("lt", 255) in c
            assert ("press", press_name) in c, c
            if kind == "x360":
                assert ("press", dpad_name) in c, c
            else:
                assert ("dpad", dpad_name) in c, c
            n = pw._pad.updates
            pw.full_update(base, 0.25, 0.0)
            assert pw._pad.updates == n
            b2 = dict(base)
            b2["buttons"] = frozenset()
            pw.full_update(b2, 0.25, 0.0)
            assert ("rel", press_name) in pw._pad.calls, pw._pad.calls
    finally:
        io.vg = old_vg
        pio.vg = old_pio_vg
        io.PadWriter._X360_MAP = io.PadWriter._DS4_MAP = io.PadWriter._DS4_DPAD = None


def t_aim_keys_engine():
    from nyosu.config import Settings
    from nyosu.engine import AimEngine

    def prof(mode, aim_keys, snap=None, keys=None):
        s = Settings()
        s.input_mode = mode
        s.aim_keys = aim_keys
        s.validate()
        e = AimEngine.__new__(AimEngine)
        e.pad_reader = None
        return e._active_profile(s, pad_snap=snap, keys=keys)

    pad = {"connected": True, "lt": 0.9, "rt": 0.9, "buttons": frozenset()}
    p, on = prof("pad", "lt", snap=pad)
    assert on and p.mode == "ads"
    p, on = prof("pad", "rt", snap=pad)
    assert on and p.mode == "hip"
    p, on = prof("pad", "rmb,lmb", snap=pad)
    assert on and p.mode == "ads"
    p, on = prof("pad", "lb", snap={**pad, "buttons": frozenset({"lb"})})
    assert on and p.mode == "ads"
    p, on = prof("pad", "lt", snap={**pad, "lt": 0.0, "rt": 0.0})
    assert not on
    p, on = prof("mouse", "lmb", keys={"rmb": True, "lmb": False})
    assert not on
    p, on = prof("mouse", "rmb,lmb", keys={"rmb": True, "lmb": False})
    assert on and p.mode == "ads"
    p, on = prof("converter", "lt", snap=pad, keys={"rmb": False, "lmb": False})
    assert on and p.mode == "ads"
    s = Settings()
    s.aim_keys = "lt, bogus, RMB, lt, rb"
    s.validate()
    assert s.aim_keys == "lt,rmb,rb", s.aim_keys
    s.aim_keys = "zzz"
    s.validate()
    assert s.aim_keys == "rmb,lmb,lt,rt"


def t_activation_mode():
    from nyosu.config import Settings
    from nyosu.engine import AimEngine

    def prof(mode, act, snap=None, keys=None):
        s = Settings()
        s.input_mode = mode
        s.activation_mode = act
        s.validate()
        e = AimEngine.__new__(AimEngine)
        e.pad_reader = None
        return e._active_profile(s, pad_snap=snap, keys=keys)

    pad_both = {"connected": True, "lt": 0.9, "rt": 0.9, "buttons": frozenset()}
    pad_ads = {"connected": True, "lt": 0.9, "rt": 0.0, "buttons": frozenset()}
    pad_fire = {"connected": True, "lt": 0.0, "rt": 0.9, "buttons": frozenset()}
    # either (default): aim alone or fire alone starts the AI
    p, on = prof("pad", "either", snap=pad_ads)
    assert on and p.mode == "ads"
    p, on = prof("pad", "either", snap=pad_fire)
    assert on and p.mode == "hip"
    p, on = prof("mouse", "either", keys={"rmb": True, "lmb": False})
    assert on and p.mode == "ads"
    p, on = prof("mouse", "either", keys={"rmb": False, "lmb": True})
    assert on and p.mode == "hip"
    # both: neither alone is enough, together starts with the ADS profile
    p, on = prof("pad", "both", snap=pad_ads)
    assert not on
    p, on = prof("pad", "both", snap=pad_fire)
    assert not on
    p, on = prof("pad", "both", snap=pad_both)
    assert on and p.mode == "ads"
    p, on = prof("mouse", "both", keys={"rmb": True, "lmb": False})
    assert not on
    p, on = prof("mouse", "both", keys={"rmb": True, "lmb": True})
    assert on and p.mode == "ads"
    # defaults and validation
    s = Settings()
    assert s.activation_mode == "either"
    assert set(s.aim_keys.split(",")) >= {"rmb", "lmb", "lt", "rt"}
    s.activation_mode = "bogus"
    s.validate()
    assert s.activation_mode == "either"


def t_settings_backup_restore():
    from nyosu.config import BACKUP_PATH, SETTINGS_PATH, Settings, load_settings, save_settings
    try:
        s = Settings()
        s.hip_strength = 0.71
        assert save_settings(s) is True
        s.hip_strength = 0.72
        assert save_settings(s) is True
        assert BACKUP_PATH.exists()
        SETTINGS_PATH.write_text("{corrupted!", encoding="utf-8")
        s2 = load_settings()
        assert abs(s2.hip_strength - 0.71) < 1e-9, s2.hip_strength
    finally:
        for pp in (SETTINGS_PATH, BACKUP_PATH):
            try:
                pp.unlink()
            except Exception:
                pass


def t_settings_roundtrip_random():
    import dataclasses as _dc
    import random as _rnd

    from nyosu.config import Settings, load_settings, save_settings
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(root, "apex_v8s_3class.onnx")
    try:
        for seed in range(5):
            rng = _rnd.Random(seed)
            s = Settings()
            s.model_path = model
            s.hip_strength = rng.uniform(0.1, 1.4)
            s.ads_conf = rng.uniform(0.1, 0.9)
            s.crop_size = rng.choice([320, 480, 640, 960, 1280])
            s.sticky_ms = rng.randrange(0, 1000)
            s.prediction = rng.uniform(0.0, 2.0)
            s.input_mode = rng.choice(["mouse", "pad", "converter"])
            s.activation_mode = rng.choice(["either", "both"])
            s.aim_keys = rng.choice(["rmb,lmb,lt,rt", "lt", "rmb", "lb,rb"])
            assert save_settings(s) is True
            s2 = load_settings()
            for f in _dc.fields(Settings):
                v1, v2 = getattr(s, f.name), getattr(s2, f.name)
                if isinstance(v1, float):
                    assert abs(v1 - v2) < 1e-9, (seed, f.name, v1, v2)
                else:
                    assert v1 == v2, (seed, f.name, v1, v2)
    finally:
        from nyosu.config import SETTINGS_PATH
        try:
            SETTINGS_PATH.unlink()
        except Exception:
            pass


def t_settings_backward_compat():
    import json as _json

    from nyosu.config import SETTINGS_PATH, load_settings
    old = {"hip_strength": 0.66, "model_path": "/nonexistent/deleted.onnx", "enabled": True}
    try:
        SETTINGS_PATH.write_text(_json.dumps(old), encoding="utf-8")
        s = load_settings()
        assert abs(s.hip_strength - 0.66) < 1e-9
        assert s.hide_gui_from_capture is True and s.hide_overlay_from_capture is True
        assert s.activation_mode == "either"
        assert "lt" in s.aim_keys and "rt" in s.aim_keys
    finally:
        try:
            SETTINGS_PATH.unlink()
        except Exception:
            pass


def t_worker_gone():
    import threading as _th

    from nyosu.engine import AimEngine
    e = AimEngine.__new__(AimEngine)
    e._worker = None
    e._worker_error = None
    assert e._worker_gone() is False
    done = _th.Thread(target=lambda: None)
    done.start()
    done.join()
    e._worker = done
    assert e._worker_gone() is True
    ev = _th.Event()
    alive = _th.Thread(target=lambda: ev.wait(2.0), daemon=True)
    alive.start()
    e._worker = alive
    assert e._worker_gone() is False
    ev.set()
    alive.join(2.0)


def t_pad_disconnect_reset():
    app, w = _make_window()
    from nyosu.input_io import PadReader
    r = PadReader()
    with r._lock:
        r._state = {"connected": True, "name": "TestPad", "kind": "xbox",
                    "lx": 0.5, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.7, "rt": 0.0,
                    "buttons": frozenset({"a"})}
    r._disconnect()
    snap = r.snapshot()
    assert snap["connected"] is False and snap["lt"] == 0.0 and snap["buttons"] == frozenset()
    w.close()
    app.processEvents()


def t_gui_quick_and_perf():
    app, w = _make_window()
    assert hasattr(w, "btn_quick") and hasattr(w, "_perf")
    assert "開始" in w.btn_quick.text()
    assert w._quick_meta.text() != ""
    w._perf.push(120.0, 6.0)
    w._perf.push(118.0, 6.4)
    g = w._perf.grab()
    assert g.width() > 50
    w._on_status({"fps": 90.0, "infer_ms": 8.0, "loop_ms": 1.0, "dets": 0,
                  "target": False, "model": True, "mode": "x", "infer_fps": 90.0})
    assert abs(w._perf._fps[-1] - 90.0) < 1e-9
    w.close()
    app.processEvents()


def t_always_on_top():
    from PyQt6.QtCore import Qt

    from nyosu.config import Settings
    s = Settings()
    assert s.always_on_top is False
    s.always_on_top = "junk"
    s.validate()
    assert s.always_on_top is False
    app, w = _make_window()
    w._on_always_on_top(True)
    app.processEvents()
    assert w.settings.always_on_top is True
    assert bool(w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    w._on_always_on_top(False)
    app.processEvents()
    assert not (w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    w.close()
    app.processEvents()


def t_reset_all_settings():
    from PyQt6.QtWidgets import QMessageBox

    app, w = _make_window()
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        w.settings.hip_strength = 0.33
        w.settings.input_mode = "pad"
        w._reset_all_settings()
        assert abs(w.settings.hip_strength - 0.88) < 1e-9
        assert w.settings.input_mode == "mouse"
        assert w.engine is None
    finally:
        QMessageBox.question = orig
    w.close()
    app.processEvents()


def t_panic_clears_engine():
    app, w = _make_window()
    from nyosu.config import Settings
    from nyosu.engine import AimEngine
    e = AimEngine(Settings(), None)
    w.engine = e
    w.panic()
    app.processEvents()
    assert w.engine is None
    assert "開始" in w.btn_quick.text(), w.btn_quick.text()
    assert w._quick_state.text() == "緊急停止"
    w.close()
    app.processEvents()


def t_unified_profile():
    from nyosu.config import Settings
    s = Settings()
    s.hip_strength = 0.9
    s.hip_fov = 222
    s.ads_strength = 0.3
    s.ads_fov = 111
    s.ads_enabled = False
    s.validate()
    assert s.ads_strength == s.hip_strength and s.ads_fov == s.hip_fov
    assert s.ads_enabled is True
    pa = s.aim_profile("ads")
    ph = s.aim_profile("hip")
    assert pa.mode == "ads" and ph.mode == "hip"
    for attr in ("enabled", "strength", "smoothing", "fov", "aim_height", "conf"):
        assert getattr(pa, attr) == getattr(ph, attr), attr


def t_fov_solid_circle():
    app, w = _make_window()
    from PyQt6.QtCore import Qt as _Qt
    assert w.overlay._pen_fov.style() == _Qt.PenStyle.SolidLine
    assert w.overlay._pen_fov.dashPattern() == []
    w.stack.setCurrentIndex(3)
    app.processEvents()
    from nyosu.widgets import OverlayPreview
    pv = w.findChild(OverlayPreview)
    assert pv is not None
    assert pv._pen_fov.style() == _Qt.PenStyle.SolidLine
    assert pv._pen_fov.dashPattern() == []
    w.close()
    app.processEvents()


def t_preset_hotkey_signal():
    app, w = _make_window()
    w.hotkey_preset.emit(1)
    app.processEvents()
    assert w.settings.preset_name == "バランス", w.settings.preset_name
    assert abs(w.settings.hip_strength - 0.88) < 1e-9
    assert w.settings.ads_strength == w.settings.hip_strength
    w.close()
    app.processEvents()


def t_session_stats():
    app, w = _make_window()
    w.stack.setCurrentIndex(5)
    app.processEvents()
    assert hasattr(w, "_session_vals")
    w.engine = object()
    w._stat_start = __import__("time").time()
    d = {"fps": 100.0, "infer_ms": 5.0, "loop_ms": 1.0, "dets": 1,
         "target": True, "model": True, "mode": "hip", "infer_fps": 90.0}
    w._on_status(d)
    d2 = dict(d)
    d2["target"] = False
    w._on_status(d2)
    assert w._session_vals["lock"].text() == "50%", w._session_vals["lock"].text()
    assert ":" in w._session_vals["time"].text()
    w.engine = None
    w.close()
    app.processEvents()


def t_new_tunables_validate():
    from nyosu.config import Settings
    s = Settings()
    assert (s.aim_curve, s.x_scale, s.y_scale) == (1.0, 1.0, 1.0)
    assert (s.max_move_px, s.aim_delay_ms, s.jitter_px) == (0, 0, 0.0)
    s.aim_curve = 99.0
    s.x_scale = float("nan")
    s.max_move_px = -5
    s.aim_delay_ms = 99999
    s.jitter_px = -1.0
    s.validate()
    assert s.aim_curve == 2.0 and s.x_scale == 0.3
    assert s.max_move_px == 0 and s.aim_delay_ms == 300 and s.jitter_px == 0.0


def _dispatch_capture(s, target, cx=500.0, cy=500.0):
    import nyosu.engine as eng
    from nyosu.engine import AimEngine
    calls = []
    orig = eng.mouse_move
    eng.mouse_move = lambda x, y: calls.append((x, y))
    try:
        e = AimEngine(s, None)
        e._last_dispatch = 0.0
        e._acc_x = e._acc_y = 0.0
        prof = s.aim_profile("hip")
        e._dispatch(target, cx, cy, prof, s, firing=False)
    finally:
        eng.mouse_move = orig
    return calls


def t_dispatch_tunables():
    import math as _math

    from nyosu.config import Settings
    s = Settings()
    base = _dispatch_capture(s, (620.0, 500.0))
    assert base and base[0][0] > 0
    s2 = Settings()
    s2.x_scale = 1.5
    dbl = _dispatch_capture(s2, (620.0, 500.0))
    assert dbl[0][0] >= base[0][0] * 1.4, (base, dbl)
    s3 = Settings()
    s3.aim_curve = 2.0
    curved = _dispatch_capture(s3, (620.0, 500.0))
    assert curved[0][0] < base[0][0], (base, curved)
    s4 = Settings()
    s4.max_move_px = 5
    capped = _dispatch_capture(s4, (900.0, 900.0))
    assert capped and _math.hypot(*capped[0]) <= 7.5, capped
    s5 = Settings()
    s5.jitter_px = 3.0
    jit = _dispatch_capture(s5, (620.0, 500.0))
    assert jit and jit[0][0] > 0


def t_aim_ready_delay():
    from nyosu.config import Settings
    from nyosu.engine import AimEngine
    s = Settings()
    s.aim_delay_ms = 200
    e = AimEngine(s, None)
    e._aim_since = None
    assert e._aim_ready(True, s, 100.0) is False
    assert e._aim_ready(True, s, 100.15) is False
    assert e._aim_ready(True, s, 100.25) is True
    assert e._aim_ready(False, s, 100.30) is False
    assert e._aim_since is None
    s.aim_delay_ms = 0
    assert e._aim_ready(True, s, 200.0) is True


def t_aim_point_mode():
    import time as _time

    from nyosu.config import Settings
    from nyosu.engine import AimEngine
    s = Settings()
    assert s.aim_point_mode == "head"
    s.aim_point_mode = "junk"
    s.validate()
    assert s.aim_point_mode == "head"
    s.aim_point_mode = "custom"
    s.validate()
    assert s.aim_point_mode == "head"
    dets = [{"box": [400, 400, 500, 700], "conf": 0.9}]
    e = AimEngine(s, None)
    prof = s.aim_profile("hip")
    s.aim_point_mode = "body"
    pt = e._choose(dets, 500, 500, prof, s, now=_time.time())
    assert pt is not None and abs(pt[1] - (400 + 300 * 0.38)) < 1e-6, pt
    s.aim_point_mode = "head"
    e._sticky = None
    pt = e._choose(dets, 500, 500, prof, s, now=_time.time())
    assert pt is not None and abs(pt[1] - (400 + 300 * 0.10)) < 1e-6, pt
    small = [{"box": [450, 400, 500, 450], "conf": 0.9}]
    e._sticky = None
    pt = e._choose(small, 500, 500, prof, s, now=_time.time())
    assert pt is not None and abs(pt[1] - (400 + 50 * 0.16)) < 1e-6, pt
    s.aim_point_mode = "random"
    e._sticky = None
    e._rand_mode = None
    pt1 = e._choose(dets, 500, 500, prof, s, now=_time.time())
    m1 = e._rand_mode
    assert m1 in ("head", "body")
    pt2 = e._choose(dets, 500, 500, prof, s, now=_time.time())
    assert e._rand_mode == m1 and pt2 == pt1


def t_overlay_malformed_dets():
    app, w = _make_window()
    w.overlay.update_data(
        [{"box": "xx", "conf": "yy"}, {"box": [1, 2, 3], "conf": 0.5},
         {"box": [1, 2, 3, 4], "conf": "0.9"}, "junk", None],
        {"target": None})
    assert len(w.overlay.dets) == 1, w.overlay.dets
    assert w.overlay.dets[0]["conf"] == 0.9
    g = w.overlay.grab()
    assert g.width() > 0
    w.close()
    app.processEvents()


def t_pad_disconnect_closes():
    from nyosu.input_io import PadReader
    calls = []

    class _Joy:
        def quit(self):
            calls.append("quit")

    class _Ctrl:
        def close(self):
            calls.append("close")

    r = PadReader()
    r._joy = _Joy()
    r._ctrl = _Ctrl()
    r._disconnect()
    assert calls == ["quit", "close"], calls
    assert r._joy is None and r._ctrl is None
    assert r.snapshot()["connected"] is False


def t_padwriter_failed_update_keeps_buttons():
    import types as _types

    import nyosu.pad_io as pio

    class _E:
        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return hash(self.v)

        def __eq__(self, o):
            return getattr(o, "v", None) == self.v

    class _Pad:
        def __init__(self):
            self.calls = []
            self.fail = True

        def left_joystick(self, x_value, y_value):
            pass

        def right_joystick(self, x_value, y_value):
            pass

        def left_trigger(self, value):
            pass

        def right_trigger(self, value):
            pass

        def press_button(self, button):
            self.calls.append(("press", button.v))

        def release_button(self, button):
            self.calls.append(("rel", button.v))

        def update(self):
            if self.fail:
                raise RuntimeError("boom")

        def reset(self):
            pass

    names_x = ["XUSB_GAMEPAD_A", "XUSB_GAMEPAD_B", "XUSB_GAMEPAD_X", "XUSB_GAMEPAD_Y",
               "XUSB_GAMEPAD_LEFT_SHOULDER", "XUSB_GAMEPAD_RIGHT_SHOULDER", "XUSB_GAMEPAD_BACK",
               "XUSB_GAMEPAD_START", "XUSB_GAMEPAD_LEFT_THUMB", "XUSB_GAMEPAD_RIGHT_THUMB",
               "XUSB_GAMEPAD_DPAD_UP", "XUSB_GAMEPAD_DPAD_DOWN", "XUSB_GAMEPAD_DPAD_LEFT", "XUSB_GAMEPAD_DPAD_RIGHT"]
    old_vg = pio.vg
    pio.vg = _types.SimpleNamespace(
        XUSB_BUTTON=_types.SimpleNamespace(**{k: _E(k) for k in names_x}),
        DS4_BUTTONS=_types.SimpleNamespace(),
        DS4_DPAD_DIRECTIONS=_types.SimpleNamespace())
    pio.PadWriter._X360_MAP = None
    try:
        pw = pio.PadWriter.__new__(pio.PadWriter)
        pw.kind = "x360"
        pw._pad = _Pad()
        pw._warned = False
        pw._last_sent = None
        pw._sent_buttons = frozenset()
        state = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0,
                 "buttons": frozenset({"a"})}
        pw.full_update(state, 0.0, 0.0)
        assert pw._sent_buttons == frozenset(), pw._sent_buttons
        assert pw._last_sent is None
        pw._pad.fail = False
        pw.full_update(state, 0.0, 0.0)
        assert ("press", "XUSB_GAMEPAD_A") in pw._pad.calls
        assert pw._sent_buttons == frozenset({"a"})
    finally:
        pio.vg = old_vg
        pio.PadWriter._X360_MAP = None


def t_capture_protect_generation():
    app, w = _make_window()
    from nyosu.capture_protect import apply_capture_protection
    apply_capture_protection(w, True)
    g1 = w._cap_gen
    apply_capture_protection(w, False)
    assert w._cap_gen == g1 + 1
    apply_capture_protection(w, True)
    assert w._cap_gen == g1 + 2
    w.close()
    app.processEvents()


def t_infer_bad_frame():
    import os as _os

    import numpy as np

    from nyosu.config import Settings
    from nyosu.vision import Detector
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    s = Settings()
    s.model_path = _os.path.join(root, "apex_v8s_3class.onnx")
    d = Detector()
    d.load(s)
    assert d.session is not None
    assert d.infer(None, s, 640, 640) == []
    assert d.infer(np.zeros((100, 100), dtype=np.uint8), s, 640, 640) == []
    assert d.infer(np.zeros((100, 100, 4), dtype=np.uint8), s, 640, 640) == []


_JUNK = [None, True, False, 0, -1, 3.5, -2.5e9, 1e18, float("nan"), float("inf"),
         "", "abc", "1.5", [], [1, 2], {}, {"a": 1}, "head", "rmb"]


def t_fuzz_settings_junk():
    import json as _json
    import random as _random
    from dataclasses import asdict as _asdict
    from dataclasses import fields as _fields

    from nyosu.config import Settings
    rng = _random.Random(2026)
    names = [f.name for f in _fields(Settings)]
    for _ in range(300):
        s = Settings()
        for name in names:
            if rng.random() < 0.5:
                setattr(s, name, rng.choice(_JUNK))
        s.validate()
        _json.dumps(_asdict(s))
        s2 = Settings()
        for k, v in _json.loads(_json.dumps(_asdict(s))).items():
            setattr(s2, k, v)
        s2.validate()
        assert _asdict(s) == _asdict(s2)


def t_fuzz_tracking_finite():
    import math as _math
    import random as _random

    from nyosu.tracking import DetTracker, KalmanCV
    rng = _random.Random(7)
    tr = DetTracker()
    now = 0.0
    for _ in range(200):
        now += 0.016
        dets = []
        for _ in range(rng.randrange(0, 5)):
            vals = [rng.choice([rng.uniform(-500, 2000), float("nan"), float("inf"), -1e12])
                    for _ in range(4)]
            dets.append({"box": vals, "conf": rng.choice([0.5, float("nan"), 2.0])})
        out = tr.update(dets, now)
        for d in out:
            assert all(_math.isfinite(v) for v in d["box"]), d
            assert _math.isfinite(d["conf"]), d
    k = KalmanCV()
    tnow = 0.0
    for _ in range(200):
        tnow += rng.choice([0.0, 0.008, 0.016, 0.1, 1.0])
        pos = (rng.uniform(-1000, 3000), rng.uniform(-1000, 3000))
        est = k.update(pos, tnow)
        assert all(_math.isfinite(v) for v in est), est


def t_fuzz_postprocess():
    import math as _math

    import numpy as np

    from nyosu.config import Settings
    from nyosu.vision import Detector
    rng = np.random.default_rng(3)
    d = Detector()
    s = Settings()
    for _ in range(60):
        c = int(rng.integers(5, 9))
        n = int(rng.integers(1, 300))
        arr = rng.normal(300, 400, size=(c, n)).astype(np.float32)
        arr[4:] = rng.random((c - 4, n)).astype(np.float32)
        if rng.random() < 0.3:
            arr[0, 0] = np.nan
            arr[1, 1] = np.inf
        out = d._postprocess(arr.copy(), 1.0, 0, 0, 640, 640, s)
        for det in out:
            x1, y1, x2, y2 = det["box"]
            assert 0 <= x1 < x2 <= 639 and 0 <= y1 < y2 <= 639, det
            assert _math.isfinite(det["conf"])


def t_fuzz_standardize():
    import random as _random

    from nyosu.input_io import standardize_raw, standardize_sdl
    rng = _random.Random(11)
    for _ in range(300):
        na = rng.randrange(0, 9)
        nb = rng.randrange(0, 14)
        axes = [rng.uniform(-2, 2) for _ in range(na)]
        btns = [rng.random() < 0.3 for _ in range(nb)]
        hat = rng.choice([None, (0, 0), (1, 1), (-1, 0)])
        for kind in ("xbox", "ps"):
            st = standardize_raw(axes, btns, hat, kind, (4, 5) if rng.random() < 0.5 else None)
            for kkey in ("lx", "ly", "rx", "ry"):
                assert -1.0 <= st[kkey] <= 1.0, st
            for kkey in ("lt", "rt"):
                assert 0.0 <= st[kkey] <= 1.0, st
        axes6 = [rng.randrange(-40000, 40000) for _ in range(rng.randrange(0, 8))]
        btns15 = [rng.random() < 0.3 for _ in range(rng.randrange(0, 17))]
        st = standardize_sdl(axes6, btns15)
        assert -1.0 <= st["lx"] <= 1.0 and 0.0 <= st["lt"] <= 1.0


def t_fuzz_choose_dispatch():
    import random as _random
    import time as _time

    import nyosu.engine as eng
    from nyosu.config import Settings
    from nyosu.engine import AimEngine
    rng = _random.Random(99)
    moves = []
    orig = eng.mouse_move
    eng.mouse_move = lambda x, y: moves.append((x, y))
    try:
        for _ in range(120):
            s = Settings()
            s.strength = None
            s.hip_strength = rng.uniform(0.05, 1.5)
            s.hip_smoothing = rng.randrange(1, 21)
            s.hip_fov = rng.randrange(40, 601)
            s.hip_conf = rng.uniform(0.05, 0.95)
            s.aim_curve = rng.uniform(0.5, 2.0)
            s.x_scale = rng.uniform(0.3, 1.5)
            s.y_scale = rng.uniform(0.3, 1.5)
            s.max_move_px = rng.randrange(0, 201)
            s.jitter_px = rng.uniform(0, 5)
            s.deadzone = rng.uniform(0, 8)
            s.aim_point_mode = rng.choice(["head", "body", "random"])
            s.target_priority = rng.choice(["nearest", "biggest", "conf"])
            s.validate()
            e = AimEngine(s, None)
            prof = s.aim_profile("hip")
            dets = []
            for _ in range(rng.randrange(0, 4)):
                x1 = rng.uniform(0, 900)
                y1 = rng.uniform(0, 900)
                dets.append({"box": [x1, y1, x1 + rng.uniform(5, 200), y1 + rng.uniform(5, 300)],
                             "conf": rng.uniform(0, 1)})
            pt = e._choose(dets, 500, 500, prof, s, now=_time.time())
            if pt is not None:
                e._dispatch(pt, 500, 500, prof, s, firing=rng.random() < 0.5)
    finally:
        eng.mouse_move = orig


def t_engine_finished_cleanup():
    import time as _time

    app, w = _make_window()
    w.settings.model_path = "/nonexistent/dead.onnx"
    w.toggle_engine()
    assert w.engine is not None
    deadline = _time.time() + 10.0
    while w.engine is not None and _time.time() < deadline:
        app.processEvents()
        _time.sleep(0.05)
    app.processEvents()
    assert w.engine is None
    assert w.settings.enabled is False
    assert w.sb_state.text() == "待機中", w.sb_state.text()
    w.close()
    app.processEvents()



def main():
    import platform as _pf
    print(f"env: python {sys.version.split()[0]} / {_pf.platform()}")
    for _m in ("numpy", "cv2", "onnxruntime", "PyQt6", "mss"):
        try:
            _mod = __import__(_m)
            _v = getattr(_mod, "__version__", None)
            if _v is None and _m == "PyQt6":
                from PyQt6.QtCore import PYQT_VERSION_STR as _v
            print(f"env: {_m} {_v if _v is not None else 'installed'}")
        except Exception:
            print(f"env: {_m} unavailable")
    check("settings_junk", t_settings_junk)
    check("validate_boundaries", t_validate_boundaries)
    check("class_ids_sanitize", t_class_ids_sanitize)
    check("snapshot_isolation", t_snapshot_isolation)
    check("corrupt_settings_file", t_corrupt_settings_file)
    check("settings_roundtrip", t_settings_roundtrip)
    check("postprocess", t_postprocess)
    check("kalman", t_kalman)
    check("det_tracker", t_det_tracker)
    check("detector_models", t_detector_models)
    check("detector_bad_models", t_detector_bad_models)
    check("config_guard_and_test_detection", t_config_guard_and_test_detection)
    check("engine_cycle", t_engine_cycle)
    check("engine_worker_death", t_engine_worker_death)
    check("engine_transient_failures", t_engine_transient_failures)
    check("engine_long_run_race", t_engine_long_run_race)
    check("capture_settings", t_capture_settings)
    check("capture_protect_noop", t_capture_protect_noop)
    check("nms_equivalence", t_nms_equivalence)
    check("tracker_malformed_and_ema", t_tracker_malformed_and_ema)
    check("validate_nan_inf", t_validate_nan_inf)
    check("postprocess_guards", t_postprocess_guards)
    check("padwriter_dedupe", t_padwriter_dedupe)
    check("model_info_cache", t_model_info_cache)
    check("pad_standardize", t_pad_standardize)
    check("padwriter_full_forward", t_padwriter_full_forward)
    check("aim_keys_engine", t_aim_keys_engine)
    check("activation_mode", t_activation_mode)
    check("settings_backup_restore", t_settings_backup_restore)
    check("settings_roundtrip_random", t_settings_roundtrip_random)
    check("settings_backward_compat", t_settings_backward_compat)
    check("worker_gone", t_worker_gone)
    check("pad_disconnect_reset", t_pad_disconnect_reset)
    check("gui_quick_and_perf", t_gui_quick_and_perf)
    check("always_on_top", t_always_on_top)
    check("reset_all_settings", t_reset_all_settings)
    check("panic_clears_engine", t_panic_clears_engine)
    check("unified_profile", t_unified_profile)
    check("fov_solid_circle", t_fov_solid_circle)
    check("preset_hotkey_signal", t_preset_hotkey_signal)
    check("session_stats", t_session_stats)
    check("new_tunables_validate", t_new_tunables_validate)
    check("dispatch_tunables", t_dispatch_tunables)
    check("aim_ready_delay", t_aim_ready_delay)
    check("aim_point_mode", t_aim_point_mode)
    check("overlay_malformed_dets", t_overlay_malformed_dets)
    check("pad_disconnect_closes", t_pad_disconnect_closes)
    check("padwriter_failed_update", t_padwriter_failed_update_keeps_buttons)
    check("capture_protect_generation", t_capture_protect_generation)
    check("infer_bad_frame", t_infer_bad_frame)
    check("fuzz_settings_junk", t_fuzz_settings_junk)
    check("fuzz_tracking_finite", t_fuzz_tracking_finite)
    check("fuzz_postprocess", t_fuzz_postprocess)
    check("fuzz_standardize", t_fuzz_standardize)
    check("fuzz_choose_dispatch", t_fuzz_choose_dispatch)
    check("engine_finished_cleanup", t_engine_finished_cleanup)
    ng = [r for r in RESULTS if not r[1]]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} passed")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
