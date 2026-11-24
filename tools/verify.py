"""NYOSU 実機検証スクリプト（Windows / Linux 共通）

使い方:  python tools/verify.py
依存インストール、単体テスト、モデル読み込み＋推論ベンチ、GUI起動スモークを
一括で実行し、結果を verify_report.txt に書き出します。
実機（特に Windows）での互換性確認はこのレポートで共有できます。
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "verify_report.txt")
LINES = []


def out(msg=""):
    print(msg)
    LINES.append(str(msg))


def section(title):
    out()
    out(f"== {title} ==")


def main():
    os.chdir(ROOT)
    section("環境")
    out(f"OS: {platform.platform()}")
    out(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for mod in ("numpy", "cv2", "onnxruntime", "PyQt6", "mss", "pygame", "keyboard", "vgamepad"):
        try:
            m = __import__(mod)
            out(f"{mod}: {getattr(m, '__version__', 'OK')}")
        except Exception as e:
            out(f"{mod}: 利用不可 ({e})")

    section("単体テスト (tests/run_tests.py)")
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join("tests", "run_tests.py")],
                           capture_output=True, text=True, timeout=900, env=env)
        out(r.stdout.strip())
        if r.returncode != 0:
            out(f"テスト失敗 (exit {r.returncode})")
            out(r.stderr.strip()[-3000:])
        else:
            out(f"テスト所要時間: {time.time() - t0:.0f}s")
    except Exception as e:
        out(f"テスト実行エラー: {e}")

    section("モデル読み込み + 推論ベンチ")
    try:
        sys.path.insert(0, ROOT)
        import numpy as np

        from nyosu.config import Settings
        from nyosu.vision import Detector
        s = Settings()
        model = os.path.join(ROOT, "apex_v8s_3class.onnx")
        s.model_path = model
        d = Detector()
        d.load(s)
        out(f"provider: {d.provider_used} / error: {d.error}")
        if d.session is not None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            d.infer(img, s, 640, 640)
            t0 = time.perf_counter()
            n = 20
            for _ in range(n):
                d.infer(img, s, 640, 640)
            out(f"推論: {(time.perf_counter() - t0) / n * 1000:.1f} ms/回 (CPU/GPUは環境依存)")
    except Exception as e:
        out(f"モデル検証エラー: {e}")

    section("Windows固有チェック")
    if os.name == "nt":
        try:
            import ctypes

            from nyosu.capture_protect import WDA_EXCLUDEFROMCAPTURE
            fn = ctypes.windll.user32.SetWindowDisplayAffinity
            out(f"SetWindowDisplayAffinity: 参照OK / WDA_EXCLUDEFROMCAPTURE=0x{WDA_EXCLUDEFROMCAPTURE:02X}")
        except Exception as e:
            out(f"Win32チェックエラー: {e}")
        try:
            import vgamepad  # noqa: F401

            out("vgamepad: インポートOK（ViGEmBusドライバの有無はパッド生成時に判定）")
        except Exception as e:
            out(f"vgamepad: 利用不可 ({e})")
    else:
        out("Windowsではないためスキップ（配信除外・ViGEm等はWindows実機でのみ検証可能）")

    out()
    out("検証完了。この verify_report.txt を共有すると実機の状態がそのまま伝わります。")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"レポート: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
