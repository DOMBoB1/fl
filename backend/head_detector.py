import os
import io
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image

import config

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "head-detection-vornn")
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")


class HeadDetector:
    """
    Wrapper around a hosted head-detection model.
    Returns bounding boxes as (x1, y1, x2, y2) in image coordinates.
    """

    def __init__(self) -> None:
        self.enabled = bool(ROBOFLOW_API_KEY)

    def _reject_box(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 1 or bh <= 1:
            return True

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area_ratio = (bw * bh) / float(w * h + 1e-9)

        if cy < config.HEAD_MIN_CENTER_Y_RATIO * h:
            return True
        if cy > config.HEAD_MAX_CENTER_Y_RATIO * h:
            return True

        if area_ratio < config.HEAD_MIN_AREA_RATIO or area_ratio > config.HEAD_MAX_AREA_RATIO:
            return True

        ratio = bw / (bh + 1e-9)
        if ratio < config.HEAD_BOX_MIN_ASPECT or ratio > config.HEAD_BOX_MAX_ASPECT:
            return True

        edge_margin_x = config.HEAD_EDGE_MARGIN_RATIO * w
        edge_margin_y = config.HEAD_EDGE_MARGIN_RATIO * h
        if x1 < edge_margin_x and y1 < edge_margin_y:
            return True

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return True

        if getattr(config, "HEAD_REJECT_IF_TOO_BRIGHT", True):
            gray = np.mean(roi, axis=2)
            mean_v = float(np.mean(gray))
            std_v = float(np.std(gray))
            if mean_v >= config.HEAD_BRIGHT_MEAN_THRESH and std_v <= config.HEAD_BRIGHT_STD_THRESH:
                return True

        return False

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.enabled:
            return []

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        rgb = frame_bgr[..., ::-1].astype(np.uint8)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        url = (
            f"https://detect.roboflow.com/"
            f"{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"
            f"?api_key={ROBOFLOW_API_KEY}&format=json"
        )

        try:
            resp = requests.post(
                url,
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                timeout=5.0,
            )
        except requests.RequestException:
            return []

        if not resp.ok:
            return []

        data = resp.json()
        preds = data.get("predictions", [])
        boxes: List[Tuple[int, int, int, int]] = []

        for p in preds:
            x_c = float(p.get("x", 0.0))
            y_c = float(p.get("y", 0.0))
            bw = float(p.get("width", 0.0))
            bh = float(p.get("height", 0.0))

            if bw <= 0 or bh <= 0:
                continue

            x1 = int(max(0.0, x_c - bw * 0.5))
            y1 = int(max(0.0, y_c - bh * 0.5))
            x2 = int(min(float(w - 1), x_c + bw * 0.5))
            y2 = int(min(float(h - 1), y_c + bh * 0.5))

            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            box = (x1, y1, x2, y2)
            if self._reject_box(frame_bgr, box):
                continue

            boxes.append(box)

        return boxes