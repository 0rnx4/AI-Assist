from __future__ import annotations

import math
import random
import threading
import time
from typing import Optional

import mss
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .config import Settings, log
from .input_io import PadReader, PadWriter, key_down, mouse_move
from .tracking import DetTracker, KalmanCV, target_point, target_point_ratio
from .vision import Detector


class AimEngine(QThread):
    overlay_sig = pyqtSignal(list, dict)
    status_sig = pyqtSignal(dict)
    log_sig = pyqtSignal(str)

    def __init__(self, settings: Settings, pad_reader: Optional[PadReader]):
        super().__init__()
        self.settings = settings
        self.pad_reader = pad_reader
        self._evt = threading.Event()
        self.detector = Detector()
        self.pad_writer = PadWriter(settings.pad_output)
        self._sticky = None
        self._sticky_until = 0.0
        self._rand_mode = None
        self._last_dets = []
        self._last_dets_time = 0.0
        self._capture_ms = 0.0
        self._infer_ms = 0.0
        self._loop_ms = 0.0

        self._acc_x = 0.0
        self._acc_y = 0.0

        self._track_pos = None
        self._track_time = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._kalman = KalmanCV()
        self._det_tracker = DetTracker()
        self._next_infer = 0.0
        self._prev_aim_on = False
        self._aim_since = None
        self._rand_mode = None
        self._gain = 1.0
        self._prev_err_x = None
        self._flip_count = 0
        self._flip_window = 0.0
        self._slow_count = 0
        self._fresh = False
        self._last_overlay = 0.0
        self._last_dispatch = time.perf_counter()
        self._slot_lock = threading.Lock()
        self._slot = {"dets": [], "wall": 0.0, "perf": 0.0, "ver": 0, "infer_ms": 0.0, "cap_ms": 0.0}
        self._aim_on_shared = False
        self._screen = None
        self._worker = None
        self._worker_error = None

    def stop(self):
        self._evt.set()

    def run(self):
        self._evt.clear()
        self._gain = max(0.05, float(self.settings.mouse_sens))
        self._prev_err_x = None
        self._flip_count = 0
        self._slow_count = 0
        self._screen = None
        self._worker = None
        self._worker_error = None
        self._sticky = None
        self._sticky_until = 0.0
        self._last_dets = []
        self._last_dets_time = 0.0
        self._capture_ms = 0.0
        self._infer_ms = 0.0
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._track_pos = None
        self._track_time = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._kalman.reset()
        self._det_tracker.reset()
        self._fresh = False
        self._aim_on_shared = False
        self._last_dispatch = time.perf_counter()
        with self._slot_lock:
            self._slot = {"dets": [], "wall": 0.0, "perf": 0.0, "ver": 0, "infer_ms": 0.0, "cap_ms": 0.0}
        try:
            self.detector.load(self.settings.snapshot())
            if self.detector.provider_used == "CPUExecutionProvider":
                self.log_sig.emit("警告: CPU推論は重いです。軽量化のため推論上限30fps以下を推奨します")
            if self.detector.error:
                self.log_sig.emit(f"検出器: {self.detector.error}")
            else:
                self.log_sig.emit(f"モデル読み込み完了 (プロバイダ={self.detector.provider_used})")
            if self.detector.error:
                return
            if self.settings.input_mode != "mouse" and not self.pad_writer.available:
                self.log_sig.emit("パッド出力が利用できません（ViGEmBus必須）")
            self._worker = threading.Thread(target=self._infer_worker, daemon=True)
            self._worker.start()
            self._loop()
        except Exception as e:
            log.exception("エンジンがクラッシュしました")
            self.log_sig.emit(f"エンジンクラッシュ: {e}")
        finally:
            if self._worker is not None:
                self._worker.join(timeout=2.0)
            try:
                if self.settings.auto_gain and self.settings.input_mode == "mouse":
                    self.settings.mouse_sens = round(min(5.0, max(0.05, self._gain)), 3)
            except Exception:
                log.warning("感度オート調整の書き戻しに失敗しました", exc_info=True)
            self.pad_writer.reset()

    def _infer_worker(self):
        try:
            with mss.mss() as sct:
                _mons = sct.monitors
                _mi = self.settings.monitor_index if self.settings.monitor_index < len(_mons) else 1
                monitor = _mons[_mi]
                sw, sh = monitor["width"], monitor["height"]
                self._screen = (sw, sh)
                next_t = 0.0
                prev_aim = False
                _fails = 0
                while not self._evt.is_set():
                    st = self.settings.snapshot()
                    if not st.enabled or self.detector.session is None:
                        self._evt.wait(0.05)
                        continue
                    aim = self._aim_on_shared
                    if aim and not prev_aim:
                        next_t = 0.0
                    prev_aim = aim
                    now = time.perf_counter()
                    if now < next_t:
                        self._evt.wait(min(0.004, next_t - now))
                        continue
                    crop = int(max(st.crop_size, max(st.hip_fov, st.ads_fov) * 2))
                    crop = max(64, min(crop, sw, sh))
                    if crop % 2:
                        crop -= 1
                    left = int(monitor.get("left", 0)) + max(0, min(sw // 2 - crop // 2, sw - crop))
                    top = int(monitor.get("top", 0)) + max(0, min(sh // 2 - crop // 2, sh - crop))
                    off_x = left - int(monitor.get("left", 0))
                    off_y = top - int(monitor.get("top", 0))
                    region = {"left": left, "top": top, "width": crop, "height": crop}
                    try:
                        t_cap = time.perf_counter()
                        shot = sct.grab(region)
                        buf = np.frombuffer(shot.bgra, dtype=np.uint8)
                        frame = buf.reshape(shot.height, shot.width, 4)[:, :, :3]
                        cap_ms = (time.perf_counter() - t_cap) * 1000.0
                        t_inf = time.perf_counter()
                        dets_crop = self.detector.infer(frame, st, crop, crop)
                        infer_ms = (time.perf_counter() - t_inf) * 1000.0
                        for d in dets_crop:
                            b = d["box"]
                            b[0] += off_x
                            b[1] += off_y
                            b[2] += off_x
                            b[3] += off_y
                        dets = self._det_tracker.update(dets_crop, time.time())
                        with self._slot_lock:
                            self._slot = {"dets": dets, "wall": time.time(), "perf": time.perf_counter(),
                                          "ver": self._slot["ver"] + 1, "infer_ms": infer_ms, "cap_ms": cap_ms}
                        _fails = 0
                    except Exception:
                        _fails += 1
                        if _fails >= 10:
                            raise
                        self._evt.wait(0.05)
                        continue
                    if aim:
                        interval = (1.0 / st.max_infer_fps) if st.max_infer_fps > 0 else 0.0
                    else:
                        interval = 1.0 / max(1, st.idle_infer_fps)
                    next_t = (t_inf + interval) if interval > 0 else 0.0
        except Exception as e:
            self._worker_error = str(e)
            log.exception("推論ワーカーが停止しました")

    def _worker_gone(self):
        return self._worker is not None and not self._worker.is_alive()

    def _loop(self):
        while not self._evt.is_set() and not self._screen:
            if self._worker_gone():
                self.log_sig.emit(f"推論ワーカーが起動できませんでした: {self._worker_error or '不明なエラー'}")
                return
            self._evt.wait(0.02)
        if self._evt.is_set():
            return
        sw, sh = self._screen
        cx, cy = sw // 2, sh // 2
        last_stat = time.perf_counter()
        frames = 0
        infer_frames = 0
        seen_ver = 0
        while not self._evt.is_set():
            t0 = time.perf_counter()
            wall_now = time.time()
            st = self.settings.snapshot()
            if st.input_mode == "pad":
                pad_snap = self.pad_reader.snapshot() if self.pad_reader is not None else None
                keys = None
            else:
                pad_snap = self.pad_reader.snapshot() if (st.input_mode == "converter" and self.pad_reader is not None) else None
                keys = {"rmb": key_down("rmb"), "lmb": key_down("lmb")}
            prof, aim_on = self._active_profile(st, pad_snap=pad_snap, keys=keys)
            self._aim_on_shared = aim_on
            if not st.enabled:
                self.overlay_sig.emit([], {})
                _nw = time.perf_counter()
                if _nw - last_stat >= 0.5:
                    if self._worker_gone():
                        self.log_sig.emit(f"推論ワーカーが停止しました: {self._worker_error or '不明なエラー'}")
                        return
                    last_stat = _nw
                    frames = 0
                    infer_frames = 0
                    self.status_sig.emit({"fps": 0.0, "infer_ms": self._infer_ms, "loop_ms": 0.0,
                                          "dets": 0, "target": False,
                                          "model": self.detector.session is not None,
                                          "mode": prof.mode, "infer_fps": 0.0})
                self._evt.wait(0.05)
                continue
            with self._slot_lock:
                slot = self._slot
            if slot["ver"] != seen_ver:
                seen_ver = slot["ver"]
                infer_frames += 1
                self._last_dets = slot["dets"]
                self._last_dets_time = slot["wall"]
                self._infer_ms = slot["infer_ms"]
                self._capture_ms = slot["cap_ms"] + max(0.0, (time.perf_counter() - slot["perf"]) * 1000.0)
                self._fresh = True
            else:
                self._fresh = False
            dets = self._last_dets
            if wall_now - self._last_dets_time > 0.2:
                dets = []
                self._last_dets = []
            raw_target = self._choose(dets, cx, cy, prof, st, now=wall_now) if (aim_on and dets) else None
            if raw_target is not None and self._fresh:
                self._adapt_gain(raw_target, cx, cy, st)
            if raw_target is not None:
                target = self._predict(raw_target, st)
            elif aim_on and self._kalman.ready and \
                    (time.perf_counter() - self._kalman.last_time) * 1000.0 <= max(st.hold_ms, self._infer_ms * 1.5):
                _lead = min((self._capture_ms + self._infer_ms) / 1000.0, 0.12) \
                    * float(st.prediction) if st.prediction > 0 else 0.0
                target = self._kalman.extrapolate(time.perf_counter(), _lead)
            else:
                target = None
            _ready = self._aim_ready(aim_on, st, time.perf_counter())
            if target is not None:
                if _ready:
                    self._dispatch(target, cx, cy, prof, st, pad_snap=pad_snap,
                                   firing=(keys["lmb"] if keys is not None else None))
            else:
                self._sticky = None
                self._rand_mode = None
                self._acc_x = self._acc_y = 0.0
                self._track_pos = None
                self._vel_x = self._vel_y = 0.0
                self._kalman.reset()
                if st.input_mode == "pad":
                    self._emit_pad(pad_snap, 0.0, 0.0)
                elif st.input_mode != "mouse":
                    self.pad_writer.right_stick(0.0, 0.0)
            _now_p = time.perf_counter()
            if _now_p - self._last_overlay >= 1.0 / max(5, st.overlay_fps):
                self._last_overlay = _now_p
                self.overlay_sig.emit(dets, {"target": target, "aim_active": aim_on})
            self._loop_ms = (time.perf_counter() - t0) * 1000.0
            frames += 1
            now = time.perf_counter()
            if now - last_stat >= 0.5:
                if self._worker_gone():
                    self.log_sig.emit(f"推論ワーカーが停止しました: {self._worker_error or '不明なエラー'}")
                    return
                _el = now - last_stat
                self.status_sig.emit({
                    "fps": frames / _el,
                    "infer_ms": self._infer_ms,
                    "loop_ms": self._loop_ms,
                    "dets": len(dets),
                    "target": target is not None,
                    "model": self.detector.session is not None,
                    "mode": prof.mode,
                    "infer_fps": infer_frames / _el,
                })
                frames = 0
                infer_frames = 0
                last_stat = now
            limit = 1.0 / 240 if aim_on else 1.0 / 120
            elapsed = time.perf_counter() - t0
            sleep = limit - elapsed
            if sleep > 0.001:
                self._evt.wait(sleep)


    def _active_profile(self, st, pad_snap=None, keys=None):
        want = {k.strip().lower() for k in str(st.aim_keys).split(",") if k.strip()} or {"rmb", "lmb"}
        ads = fire = False
        if st.input_mode != "pad":
            if keys is not None:
                rmb, lmb = keys["rmb"], keys["lmb"]
            else:
                rmb, lmb = key_down("rmb"), key_down("lmb")
            ads = rmb and "rmb" in want
            fire = lmb and "lmb" in want
        snap = pad_snap
        if snap is None and st.input_mode != "mouse" and self.pad_reader is not None:
            snap = self.pad_reader.snapshot()
        if snap is not None and snap.get("connected"):
            btns = snap.get("buttons") or frozenset()
            lt = snap.get("lt", 0.0) > 0.4 or "lt_btn" in btns
            rt = snap.get("rt", 0.0) > 0.4 or "rt_btn" in btns
            ads = ads or (lt and ("lt" in want or "rmb" in want)) or ("lb" in btns and "lb" in want)
            fire = fire or (rt and ("rt" in want or "lmb" in want)) or ("rb" in btns and "rb" in want)
        if getattr(st, "activation_mode", "either") == "both":
            if ads and fire:
                if st.ads_enabled:
                    return st.aim_profile("ads"), True
                if st.hip_enabled:
                    return st.aim_profile("hip"), True
        else:
            if ads and st.ads_enabled:
                return st.aim_profile("ads"), True
            if fire and st.hip_enabled:
                return st.aim_profile("hip"), True
        if st.aim_always_on:
            if st.hip_enabled:
                return st.aim_profile("hip"), True
            if st.ads_enabled:
                return st.aim_profile("ads"), True
        return st.aim_profile("ads" if ads else "hip"), False

    def _choose(self, dets, cx, cy, prof, st, now=None):
        if now is None:
            now = time.time()
        if st.aim_point_mode == "random" and self._rand_mode is None:
            self._rand_mode = random.choice(("head", "body"))
        if self._sticky is not None and now < self._sticky_until:
            for d in dets:
                if d["conf"] < prof.conf * 0.85:
                    continue
                sc = self._aim_point(d["box"], st)
                if math.hypot(sc[0] - self._sticky[0], sc[1] - self._sticky[1]) < 90:
                    self._sticky = sc
                    self._sticky_until = now + st.sticky_ms / 1000.0
                    return sc
            self._sticky = None
        best = None
        best_score = 1e18
        fov = float(prof.fov)
        for d in dets:
            if d["conf"] < prof.conf:
                continue
            tx, ty = self._aim_point(d["box"], st)
            dist = math.hypot(tx - cx, ty - cy)
            if dist > fov:
                continue
            if st.target_priority == "biggest":
                _bw = d["box"][2] - d["box"][0]
                _bh = d["box"][3] - d["box"][1]
                score = -(_bw * _bh) + dist * 0.5
            elif st.target_priority == "conf":
                score = -d["conf"] * 1000.0 + dist * 0.25
            else:
                score = dist - d["conf"] * 40
            if score < best_score:
                best_score = score
                best = (tx, ty)
        if best is not None and st.sticky_ms > 0:
            self._sticky = best
            self._sticky_until = now + st.sticky_ms / 1000.0
        return best

    def _aim_point(self, box, st):
        mode = st.aim_point_mode
        if mode == "random":
            mode = self._rand_mode or "head"
        if mode == "head":
            return target_point(box, "head")
        return target_point_ratio(box, 0.38)

    def _adapt_gain(self, raw_target, cx, cy, st):
        if not st.auto_gain or st.input_mode != "mouse":
            return
        err_x = raw_target[0] - cx
        now = time.perf_counter()
        if self._prev_err_x is None:
            self._prev_err_x = err_x
            return
        prev = self._prev_err_x
        self._prev_err_x = err_x
        if prev * err_x < 0 and abs(err_x) > 4:
            if now - self._flip_window > 0.5:
                self._flip_count = 0
                self._flip_window = now
            self._flip_count += 1
            self._slow_count = 0
            if self._flip_count >= 2:
                self._gain = max(0.05, self._gain * 0.85)
                self._flip_count = 0
                self._flip_window = now
            return
        if prev * err_x > 0 and abs(err_x) > 25 and abs(prev) > 1:
            shrink = 1.0 - abs(err_x) / abs(prev)
            if shrink < 0.10:
                self._slow_count += 1
                if self._slow_count >= 3:
                    self._gain = min(5.0, self._gain * 1.12)
                    self._slow_count = 0
            else:
                self._slow_count = 0

    def _predict(self, pos, st):
        now = time.perf_counter()
        if st.prediction <= 0.0:
            self._kalman.reset(pos, now)
            self._track_pos, self._track_time = pos, now
            self._vel_x = self._vel_y = 0.0
            return pos
        est = self._kalman.update(pos, now)
        self._track_pos, self._track_time = est, now
        self._vel_x, self._vel_y = self._kalman.vx, self._kalman.vy
        lead = min((self._capture_ms + self._infer_ms) / 1000.0 + 0.008, 0.12) \
            * float(st.prediction)


        _spd = math.hypot(self._vel_x, self._vel_y)
        _gate = max(0.0, min(1.0, (_spd - 40.0) / 160.0))
        return (est[0] + self._vel_x * lead * _gate, est[1] + self._vel_y * lead * _gate)

    def _emit_pad(self, pad_snap, sx, sy):
        base = pad_snap if (pad_snap is not None and pad_snap.get("connected")) else None
        try:
            self.pad_writer.full_update(base, sx, sy)
        except Exception:
            try:
                self.pad_writer.right_stick(sx, sy)
            except Exception:
                pass

    def _aim_ready(self, aim_on, st, now):
        if not aim_on:
            self._aim_since = None
            return False
        if self._aim_since is None:
            self._aim_since = now
        return (now - self._aim_since) * 1000.0 >= float(st.aim_delay_ms)

    def _dispatch(self, target, cx, cy, prof, st, pad_snap=None, firing=None):
        tx, ty = target
        if st.jitter_px > 0:
            tx += random.uniform(-st.jitter_px, st.jitter_px)
            ty += random.uniform(-st.jitter_px, st.jitter_px)
        dx = tx - cx
        dy = ty - cy

        _now = time.perf_counter()
        _dt = min(0.05, max(0.001, _now - self._last_dispatch))
        self._last_dispatch = _now
        _tau = max(0.01, float(prof.smoothing) * 0.014)
        _gain = 1.0 - math.exp(-_dt / _tau)
        mx = dx * _gain
        my = dy * _gain
        if math.hypot(dx, dy) < st.deadzone:
            mx = 0.0
            my = 0.0
        _dist = math.hypot(dx, dy)
        _cfac = 1.0
        if st.aim_curve != 1.0 and _dist > 1e-6:
            _cfac = max(0.05, min(20.0, (_dist / max(1.0, float(prof.fov))) ** (st.aim_curve - 1.0)))
        mx = mx * _cfac * st.x_scale
        my = my * _cfac * st.y_scale
        if firing is None:
            if st.input_mode != "pad":
                firing = key_down("lmb")
            else:
                firing = False
                _snap = pad_snap
                if _snap is None and self.pad_reader is not None:
                    _snap = self.pad_reader.snapshot()
                if _snap is not None:
                    firing = bool(_snap.get("connected")) and (_snap.get("rt", 0.0) > 0.4 or "rt_btn" in (_snap.get("buttons") or frozenset()))
        rx = st.recoil_x * _dt if (firing and st.recoil_enabled) else 0.0
        ry = st.recoil_y * _dt if (firing and st.recoil_enabled) else 0.0
        if st.input_mode == "mouse":
            sens = max(0.01, getattr(self, "_gain", st.mouse_sens) if st.auto_gain else st.mouse_sens) * prof.strength


            total_x = mx * sens + rx + self._acc_x
            total_y = my * sens + ry + self._acc_y
            sent_x = int(round(total_x))
            sent_y = int(round(total_y))
            self._acc_x = max(-1.0, min(1.0, total_x - sent_x))
            self._acc_y = max(-1.0, min(1.0, total_y - sent_y))
            if st.max_move_px > 0:
                _mag = math.hypot(sent_x, sent_y)
                if _mag > st.max_move_px:
                    _r = st.max_move_px / _mag
                    sent_x = int(round(sent_x * _r))
                    sent_y = int(round(sent_y * _r))
                    self._acc_x = 0.0
                    self._acc_y = 0.0
            if sent_x or sent_y:
                mouse_move(float(sent_x), float(sent_y))
        else:

            norm = max(1.0, float(prof.fov))
            sens = max(0.01, st.pad_sens) * prof.strength
            if math.hypot(dx, dy) < st.deadzone:
                sx = 0.0
                sy = 0.0
            else:
                sx = dx / norm * sens * 2
                sy = -dy / norm * sens * 2
            sx = sx * _cfac * st.x_scale
            sy = sy * _cfac * st.y_scale
            sx = max(-1.0, min(1.0, sx + rx / norm * sens * 2))
            sy = max(-1.0, min(1.0, sy - ry / norm * sens * 2))
            if st.input_mode == "pad":
                self._emit_pad(pad_snap, sx, sy)
            else:
                self.pad_writer.right_stick(sx, sy)
