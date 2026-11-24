from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QVBoxLayout, QWidget, QFrame,
)

from .input_io import vg
from .widgets import OverlayPreview, PerfGraph


class PagesMixin:
    def _page_dashboard(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        qc, ql = self._card("クイック操作", "エンジンの開始 / 停止はここからでもできます（ホットキーでも切替可能）")
        qrow = QHBoxLayout()
        qrow.setSpacing(14)
        self.btn_quick = QPushButton("▶ 開始")
        self.btn_quick.setObjectName("QuickPower")
        self.btn_quick.setFixedHeight(46)
        self.btn_quick.clicked.connect(self.toggle_engine)
        qrow.addWidget(self.btn_quick)
        qtexts = QVBoxLayout()
        qtexts.setSpacing(2)
        self._quick_state = QLabel("待機中")
        self._quick_state.setObjectName("QuickState")
        qtexts.addWidget(self._quick_state)
        self._quick_meta = QLabel(f"モデル: {Path(self.settings.model_path).name if self.settings.model_path else '未設定'} · プリセット: {self.settings.preset_name} · 入力: {self.settings.input_mode}")
        self._quick_meta.setObjectName("Muted")
        self._quick_meta.setWordWrap(True)
        qtexts.addWidget(self._quick_meta)
        qrow.addLayout(qtexts, 1)
        ql.addLayout(qrow)
        lay.addWidget(qc)
        lay.addWidget(self._profile_panel("エイムアシスト（腰撃ち / ADS 共通）", "hip", "エイムアシストを有効化", "腰撃ちとADSで同じ設定を使います"))
        lay.addStretch()
        return page

    def _page_tracking(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c, l = self._card("予測・捕捉")
        self._aim_enabled_toggle = self._toggle(
            l, "エイムアシスト有効", self.settings.enabled,
            lambda v: setattr(self.settings, "enabled", v))
        self._toggle(l, "常時吸い付き（キー不要）", self.settings.aim_always_on,
                     lambda v: setattr(self.settings, "aim_always_on", v))
        cols = QHBoxLayout()
        cols.setSpacing(16)
        for make in [
            lambda lay2: self._slider(lay2, "移動予測", 0, 200, int(self.settings.prediction * 100), " %", lambda v: setattr(self.settings, "prediction", v / 100.0), ""),
            lambda lay2: self._slider(lay2, "中央クロップ", 320, 1280, self.settings.crop_size, " px", lambda v: setattr(self.settings, "crop_size", v), "推論する中央範囲"),
            lambda lay2: self._slider(lay2, "スティッキー保持", 0, 500, self.settings.sticky_ms, " ms", lambda v: setattr(self.settings, "sticky_ms", v), ""),
        ]:
            colw = QWidget()
            coll = QVBoxLayout(colw)
            coll.setContentsMargins(0, 0, 0, 0)
            make(coll)
            cols.addWidget(colw)
        l.addLayout(cols)
        self._combo_map(l, "優先ターゲット", [("最も近い", "nearest"), ("最も大きい", "biggest"), ("信頼度が高い", "conf")],
                        self.settings.target_priority, lambda v: setattr(self.settings, "target_priority", v))
        lay.addWidget(c)
        c2, l2 = self._card("リコイル・安定化")
        self._toggle(l2, "リコイル補正を有効化", self.settings.recoil_enabled,
                     lambda v: setattr(self.settings, "recoil_enabled", v),
                     "OFFにすると縦・横の補正量が両方無効になります")
        cols2 = QHBoxLayout()
        cols2.setSpacing(16)
        for make in [
            lambda lay2: self._slider(lay2, "リコイル補正（縦）", 0, 400, int(self.settings.recoil_y), " px/秒", lambda v: setattr(self.settings, "recoil_y", float(v)), ""),
            lambda lay2: self._slider(lay2, "リコイル補正（横）", -200, 200, int(self.settings.recoil_x), " px/秒", lambda v: setattr(self.settings, "recoil_x", float(v)), ""),
            lambda lay2: self._slider(lay2, "デッドゾーン", 0, 80, int(self.settings.deadzone * 10), "", lambda v: setattr(self.settings, "deadzone", v / 10.0), "", lambda v: f"{v / 10:.1f}px"),
        ]:
            colw = QWidget()
            coll = QVBoxLayout(colw)
            coll.setContentsMargins(0, 0, 0, 0)
            make(coll)
            cols2.addWidget(colw)
        l2.addLayout(cols2)
        lay.addWidget(c2)
        c3, l3 = self._card("応答の調整")
        cols3 = QHBoxLayout()
        cols3.setSpacing(16)
        for make in [
            lambda lay2: self._slider(lay2, "最大移動速度", 0, 200, self.settings.max_move_px, "", lambda v: setattr(self.settings, "max_move_px", v), "1回の移動量の上限（0で無制限）", lambda v: "無制限" if v == 0 else f"{v}px"),
            lambda lay2: self._slider(lay2, "エイム開始ディレイ", 0, 300, self.settings.aim_delay_ms, " ms", lambda v: setattr(self.settings, "aim_delay_ms", v), "狙い始めてからアシストが効くまでの遅延"),
            lambda lay2: self._slider(lay2, "ジッター", 0, 50, int(self.settings.jitter_px * 10), "", lambda v: setattr(self.settings, "jitter_px", v / 10.0), "狙点に足す微小なランダム揺れ", lambda v: f"{v / 10:.1f}px"),
        ]:
            colw = QWidget()
            coll = QVBoxLayout(colw)
            coll.setContentsMargins(0, 0, 0, 0)
            make(coll)
            cols3.addWidget(colw)
        l3.addLayout(cols3)
        lay.addWidget(c3)
        lay.addStretch()
        return page

    def _page_display(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c, l = self._card("表示（オーバーレイ）", "")
        self._toggle(l, "FOV円を表示", self.settings.show_fov,
                     lambda v: setattr(self.settings, "show_fov", v))
        self._toggle(l, "検出ボックスを表示", self.settings.show_boxes,
                     lambda v: setattr(self.settings, "show_boxes", v))
        self._toggle(l, "ターゲット線を表示", self.settings.show_target_line,
                     lambda v: setattr(self.settings, "show_target_line", v))
        self._slider(l, "オーバーレイ更新", 10, 60, self.settings.overlay_fps, " fps",
                     lambda v: setattr(self.settings, "overlay_fps", v), "")
        self._toggle(l, "設定画面を常に最前面に表示", self.settings.always_on_top,
                     self._on_always_on_top, "ゲームの上に設定画面を出し続けます")
        lay.addWidget(c)
        c25, l25 = self._card("配信に映らない", "自分には見えますが、OBS / Discord / Xbox Game Bar などの画面キャプチャには映りません。Windows 10 2004以降が必要です。スクリーンショットにも映らなくなります")
        self._toggle(l25, "オーバーレイを配信に映さない", self.settings.hide_overlay_from_capture,
                     self._on_hide_overlay_capture, "FOV円・検出ボックス・ターゲット線が配信画面に出ません")
        self._toggle(l25, "設定画面（GUI）を配信に映さない", self.settings.hide_gui_from_capture,
                     self._on_hide_gui_capture, "NYOSUのウィンドウ自体が配信画面に出ません")
        lay.addWidget(c25)
        c2, l2 = self._card("プレビュー", "")
        l2.addWidget(OverlayPreview(self.settings))
        lay.addWidget(c2)
        c4, l4 = self._card("ショートカット", "")
        for keys, act in [
            (self.settings.hotkey.upper(), "エンジン開始 / 停止"),
            (self.settings.panic_key.upper(), "緊急停止"),
            ("Ctrl + 1〜6", "ページ切替"),
            ("Ctrl + Tab / Ctrl + Shift + Tab", "次 / 前のページ"),
            ("Ctrl + Alt + 1〜4", "プリセット切替（競技 / バランス / 軽量 / 遠距離）"),
        ]:
            row = QHBoxLayout()
            kl = QLabel(keys)
            kl.setObjectName("CtlLabel")
            kl.setFixedWidth(250)
            al = QLabel(act)
            al.setObjectName("Muted")
            row.addWidget(kl)
            row.addWidget(al, 1)
            l4.addLayout(row)
        lay.addWidget(c4)
        lay.addStretch()
        return page

    def _profile_panel(self, title, prefix, enable_label, enable_desc):
        f = QFrame()
        f.setObjectName("Card")
        l = QVBoxLayout(f)
        l.setContentsMargins(18, 16, 18, 16)
        l.setSpacing(10)
        hdr = QHBoxLayout()
        bar = QFrame()
        bar.setFixedSize(3, 14)
        bar.setStyleSheet("background:#D84040;border:none;border-radius:2px;")
        hdr.addWidget(bar)
        t = QLabel(title)
        t.setObjectName("CardTitle")
        hdr.addWidget(t)
        hdr.addStretch()
        l.addLayout(hdr)
        self._toggle(l, enable_label, getattr(self.settings, f"{prefix}_enabled"),
                     lambda v, p=prefix: setattr(self.settings, f"{p}_enabled", v), enable_desc)
        self._slider(l, "吸い付き強度", 10, 150, int(getattr(self.settings, f"{prefix}_strength") * 100), "",
                     lambda v, p=prefix: setattr(self.settings, f"{p}_strength", v / 100.0),
                     "検出へ吸い付く速さ", lambda v: f"{v / 100:.2f}")
        self._slider(l, "スムージング", 1, 20, getattr(self.settings, f"{prefix}_smoothing"), "",
                     lambda v, p=prefix: setattr(self.settings, f"{p}_smoothing", v),
                     "大きいほど滑らか、小さいほどキビキビ動きます", lambda v: f"{v}")
        self._slider(l, "FOV半径", 40, 600, getattr(self.settings, f"{prefix}_fov"), "",
                     lambda v, p=prefix: setattr(self.settings, f"{p}_fov", v),
                     "画面中央のこの円の中にいる敵だけを狙います", lambda v: f"{v}px")
        self._combo_map(l, "狙点", [("頭", "head"), ("胴体", "body"), ("ランダム（頭 / 胴体）", "random")],
                        self.settings.aim_point_mode, lambda v: setattr(self.settings, "aim_point_mode", v))
        self._slider(l, "最低信頼度", 10, 95, int(getattr(self.settings, f"{prefix}_conf") * 100), "",
                     lambda v, p=prefix: setattr(self.settings, f"{p}_conf", v / 100.0),
                     "これ未満の確からしさの検出は無視します", lambda v: f"{v / 100:.2f}")
        self._slider(l, "X軸スケール", 30, 150, int(self.settings.x_scale * 100), "",
                     lambda v: setattr(self.settings, "x_scale", v / 100.0),
                     "横方向の効き具合", lambda v: f"{v / 100:.2f}")
        self._slider(l, "Y軸スケール", 30, 150, int(self.settings.y_scale * 100), "",
                     lambda v: setattr(self.settings, "y_scale", v / 100.0),
                     "縦方向の効き具合", lambda v: f"{v / 100:.2f}")
        self._slider(l, "エイムカーブ", 50, 200, int(self.settings.aim_curve * 100), "",
                     lambda v: setattr(self.settings, "aim_curve", v / 100.0),
                     "1.00で直線 / 小さいと近距離で強く、大きいと遠距離で強く効きます", lambda v: f"{v / 100:.2f}")
        return f

    def _page_input(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        c0, l0 = self._card("吸い付き開始ボタン",
                            "ONのボタンを押している間、吸い付きが有効（OR判定）")
        self._aim_preview = QLabel("")
        self._aim_preview.setObjectName("Muted")
        l0.addWidget(self._aim_preview)
        for label, key in [
            ("ADS / 右クリック / LT", "rmb"),
            ("発火 / 左クリック / RT", "lmb"),
            ("パッド L2", "lt"),
            ("パッド R2", "rt"),
            ("パッド L1", "lb"),
            ("パッド R1", "rb"),
        ]:
            self._toggle(l0, label, key in self._aim_key_set(),
                         lambda on, k=key: self._toggle_aim_key(k, on))
        self._combo_map(l0, "AI開始の条件", [("どちらか押せば開始（狙う or 撃つ）", "either"),
                                            ("両方同時押しで開始（狙う + 撃つ）", "both")],
                        self.settings.activation_mode, lambda v: setattr(self.settings, "activation_mode", v))
        self._refresh_aim_preview()
        lay.addWidget(c0)
        c1, l1 = self._card("入力 / 出力")
        self._combo_map(l1, "入力モード", [("マウス", "mouse"), ("パッド", "pad"), ("コンバーター（キーマウ入力 → パッド出力）", "converter")],
                        self.settings.input_mode, self._on_input_mode)
        self._combo_map(l1, "出力パッド（パッド / コンバーター時）", [("Xbox 360", "x360"), ("DualShock 4", "ds4")],
                        self.settings.pad_output, lambda v: setattr(self.settings, "pad_output", v))
        self._slider(l1, "マウス感度係数", 5, 300, int(self.settings.mouse_sens * 100), "%",
                     lambda v: setattr(self.settings, "mouse_sens", v / 100.0))
        self._toggle(l1, "感度オート調整（DPI/ゲーム内感度/ADS倍率の違いを自動補正）", self.settings.auto_gain,
                     lambda v: setattr(self.settings, "auto_gain", v))
        self._slider(l1, "パッド感度", 10, 300, int(self.settings.pad_sens * 100), "%",
                     lambda v: setattr(self.settings, "pad_sens", v / 100.0))
        lbl = QLabel(f"vgamepad: {'利用可能' if vg else '未インストール'}")
        lbl.setObjectName("Muted")
        l1.addWidget(lbl)
        self._pad_status = QLabel("パッド: 未接続")
        self._pad_status.setObjectName("Muted")
        self._pad_status.setWordWrap(True)
        l1.addWidget(self._pad_status)
        note = QLabel("パッドモードでは物理パッドの全入力（両スティック・トリガー・全ボタン・十字キー）を仮想パッドへそのまま転送し、右スティックにだけ照準補正を合成します")
        note.setObjectName("CtlDesc")
        note.setWordWrap(True)
        l1.addWidget(note)
        lay.addWidget(c1)
        c2, l2 = self._card("ホットキー", "")
        self._entry(l2, "開始 / 停止キー", self.settings.hotkey,
                    lambda v: (setattr(self.settings, "hotkey", v.strip().lower() or "f2"), self._install_hotkeys()))
        self._entry(l2, "緊急停止キー", self.settings.panic_key,
                    lambda v: (setattr(self.settings, "panic_key", v.strip().lower() or "f8"), self._install_hotkeys()))
        lay.addWidget(c2)
        lay.addStretch()
        return page

    def _page_system(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        cm, lm = self._card("リアルタイム")
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        self._metric_vals = {}
        for i, (lbl, key) in enumerate([("照準FPS", "fps"), ("推論 ms", "infer_ms"), ("推論FPS", "infer_fps"), ("検出数", "dets")]):
            box = QVBoxLayout()
            box.setSpacing(1)
            v = QLabel("—")
            v.setObjectName("MetricVal")
            t = QLabel(lbl)
            t.setObjectName("MetricLbl")
            box.addWidget(v)
            box.addWidget(t)
            self._metric_vals[key] = v
            grid.addLayout(box, 0, i)
        self._session_vals = {}
        for j, (lbl, key) in enumerate([("稼働時間", "time"), ("ロック率", "lock")]):
            box = QVBoxLayout()
            box.setSpacing(1)
            v = QLabel("—")
            v.setObjectName("MetricVal")
            tt = QLabel(lbl)
            tt.setObjectName("MetricLbl")
            box.addWidget(v)
            box.addWidget(tt)
            self._session_vals[key] = v
            grid.addLayout(box, 1, j)
        lm.addLayout(grid)
        self._perf = PerfGraph()
        lm.addWidget(self._perf)
        lay.addWidget(cm)
        c1, l1 = self._card("モデル（Apex用YOLOv8 640推奨）")
        row = QHBoxLayout()
        lbl = QLabel("ONNXパス")
        lbl.setObjectName("Muted")
        self.ed_model = QLineEdit(self.settings.model_path)
        self.ed_model.editingFinished.connect(
            lambda: setattr(self.settings, "model_path", self.ed_model.text().strip()))
        btn = QPushButton("参照")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self._browse)
        row.addWidget(lbl)
        row.addWidget(self.ed_model, 1)
        row.addWidget(btn)
        l1.addLayout(row)
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        mlbl = QLabel("同梱モデル")
        mlbl.setObjectName("Muted")
        mrow.addWidget(mlbl)
        for text, fname in [("普通", "apex_v8s_3class.onnx"),
                            ("強い", "apex_v8m_accurate.onnx"),
                            ("弱い", "Apex_Legends_by_5pr_2_c2g6.onnx")]:
            mb = QPushButton(text)
            mb.setObjectName("Ghost")
            mb.clicked.connect(lambda _, f=fname: self._select_bundled_model(f))
            mrow.addWidget(mb)
        mrow.addStretch()
        l1.addLayout(mrow)
        self._entry(l1, "クラスID", self.settings.class_ids,
                    lambda v: setattr(self.settings, "class_ids", v))
        _mp = Path(self.settings.model_path) if self.settings.model_path else None
        self.model_info = QLabel(f"{_mp.name} · {_mp.stat().st_size / 1e6:.1f} MB" if _mp and _mp.exists() else "モデル未設定")
        self.model_info.setObjectName("Muted")
        self.model_info.setWordWrap(True)
        l1.addWidget(self.model_info)
        tbtn = QPushButton("テスト検出（画面を1回だけ解析）")
        tbtn.setObjectName("Ghost")
        tbtn.clicked.connect(self._test_detection)
        l1.addWidget(tbtn)
        self._refresh_model_info()
        lay.addWidget(c1)
        c3, l3 = self._card("パフォーマンス（軽量化）",
                            "推論上限0=制限なし。軽くするなら 推論上限30 / 待機時15 / オーバーレイ30 が目安")
        self._slider(l3, "推論上限（エイム中）", 0, 120, self.settings.max_infer_fps, " fps",
                     lambda v: setattr(self.settings, "max_infer_fps", v))
        self._slider(l3, "推論（待機時）", 5, 60, self.settings.idle_infer_fps, " fps",
                     lambda v: setattr(self.settings, "idle_infer_fps", v))
        self._slider(l3, "検出落ち保持", 0, 300, self.settings.hold_ms, " ms",
                     lambda v: setattr(self.settings, "hold_ms", v))
        self._slider(l3, "モニタ番号", 1, 4, self.settings.monitor_index, "",
                     lambda v: setattr(self.settings, "monitor_index", v))
        lay.addWidget(c3)
        c2, l2 = self._card("ログ")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        l2.addWidget(self.log_view)
        clr = QPushButton("ログをクリア")
        clr.setObjectName("Ghost")
        clr.clicked.connect(lambda: self.log_view.clear())
        l2.addWidget(clr)
        lay.addWidget(c2, 1)
        return page

    def _page_configs(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        c1, l1 = self._card("プリセット", "ステータスバーの 1〜4 からも切り替えできます")
        row = QHBoxLayout()
        row.setSpacing(8)
        for name in ["競技", "バランス", "軽量", "遠距離"]:
            b = QPushButton(name)
            b.setObjectName("Ghost")
            b.setCheckable(True)
            b.setChecked(self._preset_name == name)
            b.clicked.connect(lambda _, n=name: self._apply_preset(n))
            row.addWidget(b)
        row.addStretch()
        l1.addLayout(row)
        self._preset_label = QLabel(f"現在のプリセット: {self._preset_name}")
        self._preset_label.setObjectName("Muted")
        l1.addWidget(self._preset_label)
        lay.addWidget(c1)
        c3, l3 = self._card("名前付き設定", "")
        r1 = QHBoxLayout()
        self._named_name = QLineEdit()
        self._named_name.setPlaceholderText("設定名（例: ランク用）")
        sv = QPushButton("この名前で保存")
        sv.setObjectName("Ghost")
        sv.clicked.connect(self._save_named_config)
        r1.addWidget(self._named_name, 1)
        r1.addWidget(sv)
        l3.addLayout(r1)
        r2 = QHBoxLayout()
        self._named_combo = QComboBox()
        ld = QPushButton("読み込み")
        ld.setObjectName("Ghost")
        ld.clicked.connect(self._load_named_config)
        dl = QPushButton("削除")
        dl.setObjectName("Ghost")
        dl.clicked.connect(self._delete_named_config)
        r2.addWidget(self._named_combo, 1)
        r2.addWidget(ld)
        r2.addWidget(dl)
        l3.addLayout(r2)
        self._refresh_named_configs()
        lay.addWidget(c3)
        c2, l2 = self._card("設定ファイル", "変更は自動保存されます")
        row2 = QHBoxLayout()
        eb = QPushButton("書き出し")
        eb.setObjectName("Ghost")
        eb.clicked.connect(self._export_config)
        ib = QPushButton("読み込み")
        ib.setObjectName("Ghost")
        ib.clicked.connect(self._import_config)
        row2.addWidget(eb)
        row2.addWidget(ib)
        rst = QPushButton("全設定を初期値に戻す")
        rst.setObjectName("Ghost")
        rst.clicked.connect(self._reset_all_settings)
        row2.addWidget(rst)
        row2.addStretch()
        l2.addLayout(row2)
        lay.addWidget(c2)
        lay.addStretch()
        return page

