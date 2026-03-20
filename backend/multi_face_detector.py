import mediapipe as mp
import numpy as np


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2


class MultiFaceDetector:
    def __init__(self, min_conf=0.6):
        self.fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_conf,
        )

    def detect(self, frame_bgr, detect_width=640):  # pylint: disable=unused-argument
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        res = self.fd.process(rgb)

        boxes = []
        if not res or not res.detections:
            return boxes

        for det in res.detections:
            r = det.location_data.relative_bounding_box

            x1 = r.xmin * w
            y1 = r.ymin * h
            x2 = (r.xmin + r.width) * w
            y2 = (r.ymin + r.height) * h
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)

            bw = x2 - x1
            bh = y2 - y1
            if bw < 16 or bh < 16:
                continue

            ratio = bw / (bh + 1e-9)
            if ratio < 0.28 or ratio > 2.0:
                continue

            boxes.append((x1, y1, x2, y2))

        return boxes