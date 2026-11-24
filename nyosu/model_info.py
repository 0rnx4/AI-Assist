from __future__ import annotations

from pathlib import Path

MODEL_INFO_CACHE = {}


def model_info_text(model_path, ort):
    mp = Path(model_path) if model_path else None
    if mp is None or not mp.exists():
        return "モデル未設定", (mp.name if mp else None)
    try:
        key = (str(mp), mp.stat().st_mtime_ns, mp.stat().st_size)
    except Exception:
        key = None
    if key is not None and key in MODEL_INFO_CACHE:
        return MODEL_INFO_CACHE[key], mp.name
    if ort is None:
        return "onnxruntime が利用できません", mp.name
    try:
        sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        i0 = sess.get_inputs()[0]
        o0 = sess.get_outputs()[0]
        meta = sess.get_modelmeta().custom_metadata_map or {}
        names = meta.get("names", "?")
        size_mb = mp.stat().st_size / 1e6
        text = f"{mp.name} · {size_mb:.1f} MB · 入力 {i0.shape} · 出力 {o0.shape} · クラス {names}"
        if key is not None:
            if len(MODEL_INFO_CACHE) > 8:
                MODEL_INFO_CACHE.clear()
            MODEL_INFO_CACHE[key] = text
        return text, mp.name
    except Exception as e:
        return f"読み込み失敗: {e}", mp.name
