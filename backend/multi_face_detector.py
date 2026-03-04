import cv2
import mediapipe as mp

class MultiFaceDetector:
    def __init__(self, min_conf=0.6):
        self.fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_conf
        )

    def detect(self, frame_bgr, detect_width=640):
        """
        Detect faces on resized frame for speed.
        Returns list of boxes (x1,y1,x2,y2) in original coords.
        """
        H, W = frame_bgr.shape[:2]
        scale = detect_width / float(W)
        new_w = detect_width
        new_h = int(H * scale)

        small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self.fd.process(rgb)

        boxes = []
        if not res.detections:
            return boxes

        for det in res.detections:
            r = det.location_data.relative_bounding_box
            sx1 = int(max(0.0, r.xmin) * new_w)
            sy1 = int(max(0.0, r.ymin) * new_h)
            sx2 = int(min(1.0, r.xmin + r.width) * new_w)
            sy2 = int(min(1.0, r.ymin + r.height) * new_h)

            x1 = int(sx1 / scale)
            y1 = int(sy1 / scale)
            x2 = int(sx2 / scale)
            y2 = int(sy2 / scale)

            if (x2 - x1) >= 50 and (y2 - y1) >= 50:
                boxes.append((x1, y1, x2, y2))

        return boxes
