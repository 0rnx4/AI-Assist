from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Settings

try:
    import cv2
except Exception:
    cv2 = None
try:
    import onnxruntime as ort
except Exception:
    ort = None

from .config import log


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw = min(int(round(w * r)), size)
    nh = min(int(round(h * r)), size)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img

    left = (size - nw) // 2
    top = (size - nh) // 2
    right = size - nw - left
    bottom = size - nh - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(xyxy) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thr]
    return np.array(keep, dtype=np.int64)


def nms_fast(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if cv2 is not None and len(xyxy) > 0:
        try:
            xyxy32 = np.ascontiguousarray(xyxy, dtype=np.float32)
            boxes = np.empty((len(xyxy32), 4), dtype=np.float32)
            boxes[:, 0] = xyxy32[:, 0]
            boxes[:, 1] = xyxy32[:, 1]
            boxes[:, 2] = xyxy32[:, 2] - xyxy32[:, 0]
            boxes[:, 3] = xyxy32[:, 3] - xyxy32[:, 1]
            idx = cv2.dnn.NMSBoxes(boxes, np.ascontiguousarray(scores, dtype=np.float32),
                                   0.0, float(iou_thr))
            if idx is not None and len(idx):
                return np.asarray(idx, dtype=np.int64).flatten()
            return np.empty(0, dtype=np.int64)
        except Exception:
            pass
    return nms(xyxy, scores, iou_thr)


class Detector:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = None
        self.error = None
        self.provider_used = None
        self.input_size = 640
        self.input_dtype = np.float32
        self._wanted_key = None
        self._wanted_arr = None

    def load(self, s):
        self.session = None
        self.error = None
        if ort is None:
            self.error = "onnxruntimeが利用できません"
            return
        if cv2 is None:
            self.error = "opencvが利用できません"
            return
        p = Path(s.model_path)
        if not p.exists() or p.suffix.lower() != ".onnx":
            self.error = "モデルパスが無効です"
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = max(1, min(os.cpu_count() or 4, 4))
            opts.inter_op_num_threads = 1
            try:
                opts.enable_mem_pattern = True
                opts.enable_cpu_mem_arena = True
            except Exception:
                pass
            avail = ort.get_available_providers()
            prefs = ["OpenVINOExecutionProvider", "CUDAExecutionProvider",
                     "DmlExecutionProvider", "CPUExecutionProvider"]
            providers = [x for x in prefs if x in avail] or avail
            self.session = ort.InferenceSession(str(p), sess_options=opts, providers=providers)
            _in0 = self.session.get_inputs()[0]
            _shape = _in0.shape
            if len(_shape) == 4 and isinstance(_shape[2], int) and _shape[2] > 0:
                self.input_size = int(_shape[2])
            else:
                self.input_size = 640
            self.input_dtype = np.float16 if "float16" in str(_in0.type) else np.float32
            try:
                _dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=self.input_dtype)
                self.session.run(None, {_in0.name: _dummy})
            except Exception as e:
                log.warning("モデルのウォームアップ推論に失敗しました: %s", e)
            self.input_name = _in0.name
            self.output_names = [o.name for o in self.session.get_outputs()]
            got = self.session.get_providers()
            self.provider_used = got[0] if got else None
        except Exception as e:
            self.error = f"モデル読み込み失敗: {e}"

    def infer(self, frame, s: Settings, ow, oh):
        if self.session is None or cv2 is None:
            return []
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return []
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]
        elif frame.shape[2] < 3:
            return []

        size = self.input_size
        if frame.shape[0] == size and frame.shape[1] == size:
            padded, r, left, top = frame, 1.0, 0, 0
        else:
            padded, r, left, top = letterbox(frame, size)
        blob = None
        if padded.ndim == 3 and padded.shape[2] == 3:
            try:
                blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (size, size),
                                             (0.0, 0.0, 0.0), swapRB=True, crop=False)
            except Exception:
                blob = None
        if blob is None:
            blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
            blob *= 1.0 / 255.0
            blob = blob[np.newaxis]
        if self.input_dtype != np.float32:
            blob = blob.astype(self.input_dtype, copy=False)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        return self._postprocess(outs[0], r, left, top, ow, oh, s)

    def _postprocess(self, pred, r, left, top, ow, oh, s):
        if pred.ndim == 3:
            pred = pred[0]
        elif pred.ndim != 2:
            return []
        if pred.size == 0 or pred.shape[1] < 5:
            return []
        if not np.isfinite(pred).all():
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[1] < 5:
            return []
        xywh = pred[:, :4]
        scores = pred[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(cls)), cls]
        if self._wanted_key != s.class_ids:
            try:
                wanted = [int(x) for x in s.class_ids.split(",") if x.strip()]
            except Exception:
                wanted = [0]
            if not wanted:
                wanted = [0]
            self._wanted_key = s.class_ids
            self._wanted_arr = np.array(wanted, dtype=np.int64)
        if len(self._wanted_arr) == 1:
            mask = (conf >= s.base_confidence() * 0.85) & (cls == self._wanted_arr[0])
        else:
            mask = (conf >= s.base_confidence() * 0.85) & np.isin(cls, self._wanted_arr)
        xywh, conf = xywh[mask], conf[mask]
        if len(xywh) == 0:
            return []

        if len(conf) > 256:
            top_idx = np.argpartition(conf, -256)[-256:]
            xywh, conf = xywh[top_idx], conf[top_idx]
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x_tl, y_tl = cx - w / 2, cy - h / 2
        xyxy = np.stack([x_tl, y_tl, x_tl + w, y_tl + h], axis=1)
        idxs = nms_fast(xyxy, conf, 0.45)
        out = []
        inv_r = 1.0 / r if r else 1.0
        ow1, oh1 = ow - 1, oh - 1
        for i in idxs:
            row = xyxy[i]
            x1 = max(0, min(ow1, (row[0] - left) * inv_r))
            y1 = max(0, min(oh1, (row[1] - top) * inv_r))
            x2 = max(0, min(ow1, (row[2] - left) * inv_r))
            y2 = max(0, min(oh1, (row[3] - top) * inv_r))
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            if ix2 > ix1 and iy2 > iy1:
                out.append({"box": [ix1, iy1, ix2, iy2],
                            "conf": float(conf[i])})
        return out
