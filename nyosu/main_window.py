from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget, QMessageBox,
)

from .config import APP_DIR, Settings, _autodetect_model, load_settings, log, save_settings
from .engine import AimEngine
from .input_io import PadReader, keyboard, pygame, vg
from .capture_protect import apply_capture_protection
from .controls import ControlsMixin
from .model_info import MODEL_INFO_CACHE as _MODEL_INFO_CACHE
from .model_info import model_info_text
from .pages import PagesMixin
from .overlay import Overlay
from .vision import Detector, cv2, ort



class MainWindow(QMainWindow, PagesMixin, ControlsMixin):
    hotkey_toggle = pyqtSignal()
    hotkey_panic = pyqtSignal()
    hotkey_preset = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.overlay = Overlay(self.settings)
        self.overlay.hide()
        self.engine = None
        self.pad_reader = None
        self._hotkey_handles = []
        self._retired_threads = []
        self._stat_start = None
        self._stat_ticks = 0
        self._stat_lock = 0
        self._drag_pos = None
        self._aim_enabled_toggle = None
        self._preset_name = self.settings.preset_name

        self.setWindowTitle("NYOSU")
        self.resize(1040, 780)
        self.setMinimumSize(940, 660)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._build_ui()
        self._apply_window_flags()
        self._wire_signals()
        self._install_hotkeys()
        self._start_pad_reader()
        self._log("NYOSU 起動完了")
        self._self_test()
        self._autosave_snapshot = json.dumps(vars(self.settings), sort_keys=True, ensure_ascii=False)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start(2000)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._wrap_page(self._page_dashboard()))
        self.stack.addWidget(self._wrap_page(self._page_tracking()))
        self.stack.addWidget(self._wrap_page(self._page_input()))
        self.stack.addWidget(self._wrap_page(self._page_display()))
        self.stack.addWidget(self._wrap_page(self._page_configs()))
        self.stack.addWidget(self._wrap_page(self._page_system()))
        self.stack.currentChanged.connect(lambda i: self.nav.button(i).setChecked(True) if self.nav.button(i) else None)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_statusbar())

    def _build_sidebar(self):
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(172)
        v = QVBoxLayout(bar)
        v.setContentsMargins(10, 12, 10, 12)
        v.setSpacing(4)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for i, t in enumerate(["メイン", "追尾", "入力", "表示", "コンフィグ", "システム"]):
            b = QPushButton(t)
            b.setObjectName("NavBtn")
            b.setCheckable(True)
            b.setFixedHeight(42)
            self.nav.addButton(b, i)
            b.setToolTip(f"{t} (Ctrl+{i + 1})")
            b.clicked.connect(lambda _, idx=i: self.stack.setCurrentIndex(idx))
            v.addWidget(b)
            if i == 0:
                b.setChecked(True)
        v.addStretch()
        cap = QLabel("使用モデル")
        cap.setObjectName("SectionLbl")
        v.addWidget(cap)
        self._title_model = QLabel("—")
        self._title_model.setObjectName("Muted")
        self._title_model.setWordWrap(True)
        v.addWidget(self._title_model)
        return bar

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("Header")
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(12)
        logo = QLabel("NYOSU")
        logo.setObjectName("Logo")
        h.addWidget(logo)
        h.addStretch()
        self.btn_power = QPushButton("⏻")
        self.btn_power.setObjectName("Power")
        self.btn_power.setToolTip(f"開始 / 停止（{self.settings.hotkey.upper()}） · 緊急停止（{self.settings.panic_key.upper()}）")
        self.btn_power.setFixedSize(38, 38)
        self.btn_power.clicked.connect(self.toggle_engine)
        h.addWidget(self.btn_power)
        bmin = QPushButton("–")
        bmin.setObjectName("WinBtn")
        bmin.clicked.connect(self.showMinimized)
        h.addWidget(bmin)
        bcl = QPushButton("✕")
        bcl.setObjectName("WinBtn")
        bcl.clicked.connect(self.close)
        h.addWidget(bcl)
        return bar

    def _build_statusbar(self):
        bar = QFrame()
        bar.setObjectName("StatusBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 9, 16, 9)
        h.setSpacing(8)
        self._state_dot = QLabel("")
        self._state_dot.setObjectName("StateDot")
        self._state_dot.setFixedSize(10, 10)
        h.addWidget(self._state_dot)
        self.sb_left = QLabel("FPS — · 推論 —")
        self.sb_left.setObjectName("StatusText")
        h.addWidget(self.sb_left)
        h.addStretch()
        pl = QLabel("プリセット")
        pl.setObjectName("SectionLbl")
        h.addWidget(pl)
        self._preset_btns = {}
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for i, name in enumerate(["競技", "バランス", "軽量", "遠距離"], 1):
            b = QPushButton(str(i))
            b.setObjectName("PresetNum")
            b.setCheckable(True)
            b.setChecked(self.settings.preset_name == name)
            b.clicked.connect(lambda _, n=name: self._apply_preset(n))
            grp.addButton(b)
            self._preset_btns[name] = b
            h.addWidget(b)
        h.addSpacing(14)
        self.sb_state = QLabel("待機中")
        self.sb_state.setObjectName("StatusText")
        h.addWidget(self.sb_state)
        self.sb_mode = QLabel("腰撃ち + ADS")
        self.sb_mode.setObjectName("StatusMode")
        h.addWidget(self.sb_mode)
        return bar






    def _wrap_page(self, page):
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(page)
        return sc

    def _set_state(self, text, color):
        if hasattr(self, "sb_state"):
            self.sb_state.setText(text)
            self.sb_state.setStyleSheet(f"color:{color};font-family:Consolas,monospace;font-size:12px;")
        if hasattr(self, "_state_dot"):
            self._state_dot.setStyleSheet(f"background:{color};border-radius:5px;")
        if hasattr(self, "_quick_state"):
            self._quick_state.setText(text)
        if hasattr(self, "btn_quick"):
            want = "■ 停止" if self.engine is not None else "▶ 開始"
            if self.btn_quick.text() != want:
                self.btn_quick.setText(want)
        if hasattr(self, "_quick_meta"):
            self._quick_meta.setText(f"モデル: {Path(self.settings.model_path).name if self.settings.model_path else '未設定'} · プリセット: {self.settings.preset_name} · 入力: {self.settings.input_mode}")


    def _apply_preset(self, name):
        presets = {
            "競技": {"hip_strength": 0.95, "hip_smoothing": 4, "hip_fov": 200, "hip_aim_height": 0.18, "hip_conf": 0.50, "ads_strength": 0.80, "ads_smoothing": 5, "ads_fov": 170, "ads_aim_height": 0.15, "ads_conf": 0.60, "prediction": 1.0, "crop_size": 640, "sticky_ms": 80, "max_infer_fps": 0, "idle_infer_fps": 20, "overlay_fps": 30, "hold_ms": 120},
            "バランス": {"hip_strength": 0.88, "hip_smoothing": 5, "hip_fov": 183, "hip_aim_height": 0.20, "hip_conf": 0.45, "ads_strength": 0.65, "ads_smoothing": 6, "ads_fov": 151, "ads_aim_height": 0.16, "ads_conf": 0.60, "prediction": 0.9, "crop_size": 640, "sticky_ms": 80, "max_infer_fps": 45, "idle_infer_fps": 20, "overlay_fps": 30, "hold_ms": 120},
            "軽量": {"hip_strength": 0.85, "hip_smoothing": 6, "hip_fov": 183, "hip_aim_height": 0.20, "hip_conf": 0.45, "ads_strength": 0.60, "ads_smoothing": 7, "ads_fov": 151, "ads_aim_height": 0.16, "ads_conf": 0.60, "prediction": 0.8, "crop_size": 640, "sticky_ms": 80, "max_infer_fps": 30, "idle_infer_fps": 15, "overlay_fps": 30, "hold_ms": 150},
            "遠距離": {"hip_strength": 0.80, "hip_smoothing": 6, "hip_fov": 150, "hip_aim_height": 0.18, "hip_conf": 0.50, "ads_strength": 0.60, "ads_smoothing": 7, "ads_fov": 130, "ads_aim_height": 0.14, "ads_conf": 0.65, "prediction": 0.8, "crop_size": 480, "sticky_ms": 100, "max_infer_fps": 0, "idle_infer_fps": 20, "overlay_fps": 30, "hold_ms": 120},
        }
        cfg = presets.get(name)
        if cfg is None:
            return
        for k, v in cfg.items():
            setattr(self.settings, k, v)
        self.settings.validate()
        self._preset_name = name
        self.settings.preset_name = name
        if hasattr(self, "_preset_btns"):
            for n, b in self._preset_btns.items():
                b.setChecked(n == name)
        self._rebuild_pages()
        save_settings(self.settings)
        self._autosave_snapshot = json.dumps(vars(self.settings), sort_keys=True, ensure_ascii=False)
        self._log(f"プリセット適用: {name}")


    def _rebuild_pages(self):
        idx = self.stack.currentIndex()
        log_text = ""
        if hasattr(self, "log_view"):
            try:
                log_text = self.log_view.toPlainText()
            except Exception:
                log_text = ""
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()
        self.stack.addWidget(self._wrap_page(self._page_dashboard()))
        self.stack.addWidget(self._wrap_page(self._page_tracking()))
        self.stack.addWidget(self._wrap_page(self._page_input()))
        self.stack.addWidget(self._wrap_page(self._page_display()))
        self.stack.addWidget(self._wrap_page(self._page_configs()))
        self.stack.addWidget(self._wrap_page(self._page_system()))
        self.stack.setCurrentIndex(idx)
        if log_text and hasattr(self, "log_view"):
            self.log_view.setPlainText(log_text)


    def _autosave_tick(self):
        if hasattr(self, "_pad_status"):
            try:
                if self.pad_reader is not None:
                    _snap = self.pad_reader.snapshot()
                    if _snap.get("connected"):
                        _btns = _snap.get("buttons") or frozenset()
                        self._pad_status.setText(
                            f"パッド: {_snap.get('name', '?')}（{_snap.get('kind', '')}） · "
                            f"L {_snap.get('lt', 0.0):.2f} / R {_snap.get('rt', 0.0):.2f} · ボタン {len(_btns)}")
                    else:
                        self._pad_status.setText("パッド: 未接続")
                else:
                    self._pad_status.setText("パッド: 読み取り停止中")
            except Exception:
                pass
        try:
            cur = json.dumps(vars(self.settings), sort_keys=True, ensure_ascii=False)
        except Exception as e:
            if not getattr(self, "_autosave_warned", False):
                self._autosave_warned = True
                log.warning("自動保存の状態取得に失敗しました: %s", e)
            return
        if cur != getattr(self, "_autosave_snapshot", None):
            save_settings(self.settings)
            try:
                self._autosave_snapshot = json.dumps(vars(self.settings), sort_keys=True, ensure_ascii=False)
            except Exception:
                self._autosave_snapshot = cur

    def _refresh_model_info(self):
        if not hasattr(self, "model_info"):
            return
        _cur = self.model_info.text()
        if getattr(self, "_mi_path", None) == self.settings.model_path and ("入力" in _cur or "未設定" in _cur):
            return
        self._mi_path = self.settings.model_path
        text, name = model_info_text(self.settings.model_path, ort)
        self.model_info.setText(text)
        if name and hasattr(self, "_title_model"):
            self._title_model.setText(name)

    def _config_dir(self):
        d = APP_DIR / "configs"
        d.mkdir(exist_ok=True)
        return d

    def _refresh_named_configs(self):
        if not hasattr(self, "_named_combo"):
            return
        cur = self._named_combo.currentText()
        self._named_combo.clear()
        names = sorted(pp.stem for pp in self._config_dir().glob("*.json"))
        self._named_combo.addItems(names)
        if cur in names:
            self._named_combo.setCurrentText(cur)

    def _save_named_config(self):
        name = self._named_name.text().strip() if hasattr(self, "_named_name") else ""
        bad = set('/\\:*?"<>|')
        if not name or any(c in bad for c in name):
            self._log("名前付き設定: 使える名前を入力してください")
            return
        try:
            (self._config_dir() / f"{name}.json").write_text(json.dumps(asdict(self.settings), indent=2, ensure_ascii=False), encoding="utf-8")
            self._refresh_named_configs()
            self._named_combo.setCurrentText(name)
            self._log(f"名前付き設定を保存しました: {name}")
        except Exception as e:
            self._log(f"名前付き設定の保存に失敗しました: {e}")

    def _load_named_config(self):
        name = self._named_combo.currentText() if hasattr(self, "_named_combo") else ""
        if not name:
            self._log("名前付き設定: 保存済みの設定がありません")
            return
        try:
            data = json.loads((self._config_dir() / f"{name}.json").read_text(encoding="utf-8"))
        except Exception as e:
            self._log(f"名前付き設定の読み込みに失敗しました: {e}")
            return
        if self._apply_config_dict(data):
            self._log(f"名前付き設定を読み込みました: {name}")

    def _delete_named_config(self):
        name = self._named_combo.currentText() if hasattr(self, "_named_combo") else ""
        if not name:
            return
        try:
            (self._config_dir() / f"{name}.json").unlink()
            self._refresh_named_configs()
            self._log(f"名前付き設定を削除しました: {name}")
        except Exception as e:
            self._log(f"名前付き設定の削除に失敗しました: {e}")

    def _apply_config_dict(self, data):
        if not isinstance(data, dict):
            self._log("設定の読み込みに失敗しました: 形式が不正です")
            return False
        known = {f.name for f in fields(Settings)}
        for k, v in data.items():
            if k in known:
                setattr(self.settings, k, v)
        self.settings.validate()
        self._preset_name = self.settings.preset_name
        if hasattr(self, "_preset_btns"):
            for n, b in self._preset_btns.items():
                b.setChecked(n == self._preset_name)
        self._rebuild_pages()
        save_settings(self.settings)
        return True


    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "設定を書き出し", "nyosu_config.json", "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(asdict(self.settings), indent=2, ensure_ascii=False), encoding="utf-8")
            self._log(f"設定を書き出しました: {path}")
        except Exception as e:
            self._log(f"書き出し失敗: {e}")

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "設定を読み込み", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            self._log(f"読み込み失敗: {e}")
            return
        if self._apply_config_dict(data):
            self._log(f"設定を読み込みました: {path}")

    def _test_detection(self):
        if not self.settings.model_path or not Path(self.settings.model_path).exists():
            self._log("テスト検出: モデルが未設定です")
            return
        try:
            import mss
            det = Detector()
            det.load(self.settings)
            if det.session is None:
                self._log(f"テスト検出: モデルの読み込みに失敗しました {det.error or ''}")
                return
            with mss.mss() as sct:
                mon = sct.monitors[min(self.settings.monitor_index, len(sct.monitors) - 1)]
                img = np.array(sct.grab(mon))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if cv2 is not None else img[:, :, :3]
            h, w = frame.shape[:2]
            crop = int(max(self.settings.crop_size, max(self.settings.hip_fov, self.settings.ads_fov) * 2))
            crop = max(320, min(crop, w, h))
            x0 = (w - crop) // 2
            y0 = (h - crop) // 2
            view = frame[y0:y0 + crop, x0:x0 + crop]
            t0 = time.time()
            dets = det.infer(view, self.settings, crop, crop)
            ms = (time.time() - t0) * 1000
            if dets:
                top = max(dets, key=lambda d: d["conf"])
                self._log(f"テスト検出: {len(dets)}件検出 · 最高信頼度 {top['conf']:.2f} · {ms:.0f} ms · {det.provider_used}")
            else:
                self._log(f"テスト検出: 検出なし · {ms:.0f} ms · {det.provider_used}（敵が映っていない画面では正常です）")
        except Exception as e:
            self._log(f"テスト検出に失敗しました: {e}")

    def _aim_key_set(self):
        return {k.strip().lower() for k in self.settings.aim_keys.split(",") if k.strip()}

    def _toggle_aim_key(self, key, on):
        s = self._aim_key_set()
        if on:
            s.add(key)
        else:
            s.discard(key)
        self.settings.aim_keys = ",".join(sorted(s))
        self._refresh_aim_preview()

    def _refresh_aim_preview(self):
        keys = self._aim_key_set()
        self._aim_preview.setText(f"有効: {', '.join(sorted(keys)) if keys else 'なし'}")

    def _on_input_mode(self, v):
        self.settings.input_mode = v
        if v == "pad" and self.pad_reader is None:
            self._start_pad_reader()

    def _wire_signals(self):
        self.hotkey_toggle.connect(self.toggle_engine)
        self.hotkey_panic.connect(self.panic)
        self.hotkey_preset.connect(lambda i: self._apply_preset(["競技", "バランス", "軽量", "遠距離"][i]))
        for i in range(6):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.stack.setCurrentIndex(idx))
        sc_next = QShortcut(QKeySequence("Ctrl+Tab"), self)
        sc_next.activated.connect(lambda: self.stack.setCurrentIndex((self.stack.currentIndex() + 1) % 6))
        sc_prev = QShortcut(QKeySequence("Ctrl+Shift+Tab"), self)
        sc_prev.activated.connect(lambda: self.stack.setCurrentIndex((self.stack.currentIndex() - 1) % 6))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "ONNXモデルを選択", "", "ONNX (*.onnx)")
        if path:
            self.ed_model.setText(path)
            self.settings.model_path = path
            self._log(f"モデル: {path}")
            self._refresh_model_info()

    def _select_bundled_model(self, fname):
        p = Path(__file__).resolve().parent.parent / fname
        if not p.exists():
            self._log(f"同梱モデルが見つかりません: {fname}")
            return
        self.ed_model.setText(str(p))
        self.settings.model_path = str(p)
        save_settings(self.settings)
        self._log(f"モデルを切り替えました: {fname}（次回エンジン開始時に適用）")
        self._refresh_model_info()


    def _start_pad_reader(self):
        if pygame is None:
            self._log("pygameが利用できません。パッド読み取りを無効化します")
            return
        self._retired_threads = [th for th in self._retired_threads if th.isRunning()]
        if self.pad_reader is not None:
            self.pad_reader.stop()
            self.pad_reader.wait(1000)
            if self.pad_reader.isRunning():
                self._retired_threads.append(self.pad_reader)
        self.pad_reader = PadReader()
        self.pad_reader.state_changed.connect(self._on_pad_state)
        self.pad_reader.start()

    def _on_pad_state(self, st):
        if st.get("error"):
            self._log(f"パッド: {st['error']}")
        elif st.get("connected"):
            self._log(f"パッド接続: {st.get('name', '?')}")

    def _install_hotkeys(self):
        if keyboard is None:
            self._log("keyboardが利用できません。ホットキーを無効化します")
            return
        for h in self._hotkey_handles:
            try:
                keyboard.remove_hotkey(h)
            except Exception:
                pass
        self._hotkey_handles.clear()
        for _key, _sig in ((self.settings.hotkey, self.hotkey_toggle), (self.settings.panic_key, self.hotkey_panic)):
            _k = (_key or "").strip()
            if not _k:
                continue
            try:
                self._hotkey_handles.append(keyboard.add_hotkey(_k, _sig.emit))
            except Exception as e:
                self._log(f"ホットキーエラー ({_k}): {e}")
        for i in range(4):
            try:
                self._hotkey_handles.append(
                    keyboard.add_hotkey(f"ctrl+alt+{i + 1}", lambda i=i: self.hotkey_preset.emit(i)))
            except Exception as e:
                self._log(f"ホットキーエラー (ctrl+alt+{i + 1}): {e}")
        if self._hotkey_handles:
            self._log(f"ホットキー登録: {self.settings.hotkey.upper()}, {self.settings.panic_key.upper()}, Ctrl+Alt+1〜4（プリセット）")

    def _log(self, msg):
        log.info(msg)
        if hasattr(self, "log_view"):
            self.log_view.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _self_test(self):
        items = [
            ("Python", sys.version.split()[0]),
            ("OS", "Windows" if os.name == "nt" else os.name),
            ("OpenCV", "OK" if cv2 else "なし"),
            ("ONNX Runtime", (f"{ort.__version__} / {', '.join(ort.get_available_providers())}") if ort else "なし"),
            ("keyboard", "OK" if keyboard else "なし"),
            ("pygame", "OK" if pygame else "なし"),
            ("vgamepad", "OK" if vg else "なし"),
            ("モデル", "設定済み" if self.settings.model_path and Path(self.settings.model_path).exists() else "未設定"),
        ]
        for k, v in items:
            self._log(f"自己診断 {k}={v}")
        if self.settings.model_path and Path(self.settings.model_path).exists() and ort is not None and cv2 is not None:
            try:
                det = Detector()
                det.load(self.settings)
                if det.error:
                    self._log(f"推論ベンチ: 読み込み失敗 {det.error}")
                else:
                    frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    det.infer(frame, self.settings, 640, 640)
                    t0 = time.perf_counter()
                    n = 10
                    for _ in range(n):
                        det.infer(frame, self.settings, 640, 640)
                    ms = (time.perf_counter() - t0) / n * 1000.0
                    self._log(f"推論ベンチ {ms:.1f} ms/回 ({det.provider_used})")
            except Exception as e:
                self._log(f"推論ベンチ失敗: {e}")


    def panic(self):
        self.settings.enabled = False
        if self._aim_enabled_toggle is not None:
            self._aim_enabled_toggle.setChecked(False)
        if self.engine is not None:
            _e = self.engine
            self.engine = None
            _e.stop()
            _e.wait(1500)
            if _e.isRunning():
                self._retired_threads.append(_e)
        if hasattr(self, "btn_power"):
            self.btn_power.setStyleSheet("")
        self._set_state("緊急停止", "#FF6B61")
        self._log("緊急停止")

    def toggle_engine(self):
        if self.engine is not None:
            _e = self.engine
            _e.stop()
            _e.wait(2000)
            if _e.isRunning():
                self._retired_threads.append(_e)
            self.engine = None
            self.overlay.hide()
            self.btn_power.setStyleSheet("")
            self._set_state("待機中", "#7E8499")
            self.settings.enabled = False
            if self._aim_enabled_toggle is not None:
                self._aim_enabled_toggle.setChecked(False)
            self._log("エンジン停止")
            return
        if self.settings.input_mode == "pad" and self.pad_reader is None:
            self._start_pad_reader()
        self.settings.enabled = True
        if self._aim_enabled_toggle is not None:
            self._aim_enabled_toggle.setChecked(True)
        self.engine = AimEngine(self.settings, self.pad_reader)
        self.engine.overlay_sig.connect(self.overlay.update_data)
        self.engine.status_sig.connect(self._on_status)
        self.engine.log_sig.connect(self._log)
        self.engine.finished.connect(lambda e=self.engine: self._on_engine_finished(e))
        self.engine.start()
        try:
            import mss as _mss_mod
            with _mss_mod.mss() as _sct:
                _mons = _sct.monitors
                _mon = _mons[self.settings.monitor_index if self.settings.monitor_index < len(_mons) else 1]
                self.overlay.setGeometry(_mon["left"], _mon["top"], _mon["width"], _mon["height"])
        except Exception:
            pass
        self.overlay.show()
        self.overlay.refresh_capture_protection()
        self.btn_power.setStyleSheet("background:#D84040;color:#FFFFFF;border-color:#D84040;")
        self._stat_start = time.time()
        self._stat_ticks = 0
        self._stat_lock = 0
        self._set_state("探索中", "#E9EBF4")
        self._log("エンジン開始")


    def _on_engine_finished(self, e):
        if self.engine is not e:
            return
        self.engine = None
        self.overlay.hide()
        if hasattr(self, "btn_power"):
            self.btn_power.setStyleSheet("")
        self.settings.enabled = False
        if self._aim_enabled_toggle is not None:
            self._aim_enabled_toggle.setChecked(False)
        self._set_state("待機中", "#7E8499")
        self._log("エンジンが終了しました")

    def _on_status(self, d):
        if self.engine is not None:
            self._stat_ticks += 1
            if d.get("target"):
                self._stat_lock += 1
            if hasattr(self, "_session_vals") and self._stat_start is not None:
                _el = int(time.time() - self._stat_start)
                self._session_vals["time"].setText(f"{_el // 60}:{_el % 60:02d}")
                self._session_vals["lock"].setText(
                    f"{(100.0 * self._stat_lock / self._stat_ticks) if self._stat_ticks else 0:.0f}%")
        if hasattr(self, "_perf"):
            try:
                self._perf.push(d.get("fps", 0.0), d.get("infer_ms", 0.0))
            except Exception:
                pass
        if hasattr(self, "_metric_vals"):
            for k, lab in self._metric_vals.items():
                v = d.get(k)
                if v is not None:
                    lab.setText(f"{v:.0f}")
        locked = bool(d.get("target"))
        mode = {"hip": "腰撃ち", "ads": "ADS"}.get(str(d.get("mode", "hip")), "腰撃ち")
        if hasattr(self, "sb_left"):
            self.sb_left.setText(f"FPS {d.get('fps', 0):.0f} · 推論 {d.get('infer_ms', 0):.0f} ms · 検出 {d.get('dets', 0)}")
        if hasattr(self, "sb_mode"):
            self.sb_mode.setText(mode if d.get("model") else "モデルなし")
        if locked:
            self._set_state("ロック中", "#30D158")
        elif d.get("model"):
            self._set_state("探索中", "#E9EBF4")
        else:
            self._set_state("モデルなし", "#FF6B61")



    def _on_always_on_top(self, v):
        self.settings.always_on_top = bool(v)
        self._apply_window_flags()
        self._log("設定画面の最前面表示: " + ("ON" if v else "OFF"))

    def _apply_window_flags(self):
        want_top = bool(getattr(self.settings, "always_on_top", False))
        flags = self.windowFlags()
        has = bool(flags & Qt.WindowType.WindowStaysOnTopHint)
        if has == want_top:
            return
        if want_top:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        if self.isVisible():
            self.show()
        self._capture_applied = None
        self._refresh_capture_protection()

    def _reset_all_settings(self):
        if QMessageBox.question(self, "全設定を初期値に戻す",
                                "全ての設定を初期値に戻します。名前付き設定は削除されません。よろしいですか？") != QMessageBox.StandardButton.Yes:
            return
        if self.engine is not None:
            self.toggle_engine()
        fresh = _autodetect_model(Settings())
        for f in fields(Settings):
            setattr(self.settings, f.name, getattr(fresh, f.name))
        self.settings.validate()
        self._preset_name = self.settings.preset_name
        if hasattr(self, "_preset_btns"):
            for n, b in self._preset_btns.items():
                b.setChecked(n == self._preset_name)
        self._rebuild_pages()
        self._apply_window_flags()
        save_settings(self.settings)
        self._autosave_snapshot = json.dumps(vars(self.settings), sort_keys=True, ensure_ascii=False)
        self._log("全設定を初期値に戻しました")

    def _refresh_capture_protection(self):
        want = bool(getattr(self.settings, "hide_gui_from_capture", True))
        if getattr(self, "_capture_applied", None) is not want:
            apply_capture_protection(self, want)
            self._capture_applied = want

    def showEvent(self, e):
        super().showEvent(e)
        self._capture_applied = None
        self._refresh_capture_protection()

    def _on_hide_gui_capture(self, v):
        self.settings.hide_gui_from_capture = v
        self._capture_applied = None
        self._refresh_capture_protection()
        self._log("配信保護: GUIを配信に映さない " + ("ON" if v else "OFF") + "（Windowsでは再表示後に反映）")

    def _on_hide_overlay_capture(self, v):
        self.settings.hide_overlay_from_capture = v
        try:
            self.overlay.refresh_capture_protection()
        except Exception:
            pass
        self._log("配信保護: オーバーレイを配信に映さない " + ("ON" if v else "OFF"))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, _e):
        self._drag_pos = None

    def closeEvent(self, e):
        try:
            self._autosave_timer.stop()
        except Exception:
            pass
        self.settings.enabled = False
        if self.engine is not None:
            _e = self.engine
            _e.stop()
            _e.wait(3000)
            if _e.isRunning():
                self._retired_threads.append(_e)
            self.engine = None
        if self.pad_reader is not None:
            self.pad_reader.stop()
            self.pad_reader.wait(2000)
            if self.pad_reader.isRunning():
                self._retired_threads.append(self.pad_reader)
            self.pad_reader = None
        for th in self._retired_threads:
            try:
                th.wait(500)
            except Exception:
                pass
        self._retired_threads.clear()
        if keyboard is not None:
            for h in self._hotkey_handles:
                try:
                    keyboard.remove_hotkey(h)
                except Exception:
                    pass
        self.overlay.close()
        save_settings(self.settings)
        e.accept()
