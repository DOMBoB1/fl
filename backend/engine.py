import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

import config
from multi_face_detector import MultiFaceDetector
from tracker_flow import FlowMultiTracker
from fatigue import perclos_from_events, fatigue_percent
from attention import gaze_ratio, attentive_from_gaze


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def lm_xy(face_lms, idx, w, h):
    p = face_lms.landmark[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def eye_aspect_ratio(face_lms, eye_idx, w, h) -> float:
    p1 = lm_xy(face_lms, eye_idx[0], w, h)
    p2 = lm_xy(face_lms, eye_idx[1], w, h)
    p3 = lm_xy(face_lms, eye_idx[2], w, h)
    p4 = lm_xy(face_lms, eye_idx[3], w, h)
    p5 = lm_xy(face_lms, eye_idx[4], w, h)
    p6 = lm_xy(face_lms, eye_idx[5], w, h)

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h1 = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * h1 + 1e-9))


@dataclass
class PersonState:
    perclos_events: List[Tuple[float, bool]] = field(default_factory=list)
    eye_closed: bool = False
    eye_closed_start: Optional[float] = None


@dataclass
class EngineStats:
    faces: int = 0
    class_avg_fatigue_pct: int = 0
    class_avg_attention_pct: int = 0
    alert_active: bool = False
    fps: float = 0.0


class MonitoringEngine:
    def __init__(self):
        self.cap = None
        self.running = False

        self.detector = MultiFaceDetector(min_conf=0.6)
        self.tracker = FlowMultiTracker(
            ttl_s=config.TRACK_TTL_S,
            iou_t=config.IOU_MATCH_T,
            dist_t=config.CENTER_DIST_T,
            min_pts=config.MIN_TRACK_POINTS
        )

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.people: Dict[int, PersonState] = {}
        self.frame_idx = 0
        self.alert_start: Optional[float] = None

        self._last_frame = None
        self._last_stats = EngineStats()
        self._last_time = time.time()
        self._fps_smooth = 0.0

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available.")
        self.running = True
        self.frame_idx = 0
        self.alert_start = None
        self._last_time = time.time()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _update_fps(self, now: float):
        dt = now - self._last_time
        self._last_time = now
        fps_inst = (1.0 / dt) if dt > 1e-6 else 0.0
        self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps_inst

    def _process_frame_inplace(self, frame_bgr: np.ndarray, now: float, want_boxes: bool) -> np.ndarray:        self.tracker.update_optical_flow(frame_bgr, now)

        if self.frame_idx % config.DETECT_EVERY_N_FRAMES == 0:
            dets = self.detector.detect(frame_bgr, detect_width=config.DETECT_WIDTH)
            self.tracker.sync_with_detections(frame_bgr, dets, now)

        self.tracker.drop_stale(now)
        tracks = self.tracker.get_tracks(frame_bgr)

        for tid in tracks.keys():
            if tid not in self.people:
                self.people[tid] = PersonState()

        active_fatigue: List[int] = []
        active_attention: List[int] = []

        for tid, (x1, y1, x2, y2) in tracks.items():
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_small = cv2.resize(
                face, (config.CROP_SIZE, config.CROP_SIZE),
                interpolation=cv2.INTER_LINEAR
            )
            rgb = cv2.cvtColor(face_small, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)

            if not res.multi_face_landmarks:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame_bgr, f"ID{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )
                continue

            lms = res.multi_face_landmarks[0]

            ear_l = eye_aspect_ratio(lms, LEFT_EYE, config.CROP_SIZE, config.CROP_SIZE)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE, config.CROP_SIZE, config.CROP_SIZE)
            ear = (ear_l + ear_r) / 2.0
            is_closed = ear < config.EYES_CLOSED_EAR_T

            st = self.people[tid]
            st.perclos_events.append((now, is_closed))
            st.perclos_events = [(t, c) for (t, c) in st.perclos_events if t >= now - (config.PERCLOS_WINDOW_S + 2)]

            long_closure = False
            if is_closed and not st.eye_closed:
                st.eye_closed = True
                st.eye_closed_start = now
            elif (not is_closed) and st.eye_closed:
                st.eye_closed = False
                st.eye_closed_start = None

            if st.eye_closed and st.eye_closed_start is not None:
                if (now - st.eye_closed_start) * 1000.0 >= config.DROWSY_CLOSED_MS:
                    long_closure = True

            perclos = perclos_from_events(st.perclos_events, now, config.PERCLOS_WINDOW_S)
            fat_pct = fatigue_percent(perclos, long_closure)
            active_fatigue.append(fat_pct)

            gz = gaze_ratio(lms, config.CROP_SIZE, config.CROP_SIZE)
            att_ok = attentive_from_gaze(gz, config.LOOK_MIN, config.LOOK_MAX)
            active_attention.append(att_ok)

            color = (0, 255, 0) if att_ok else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_bgr, f"ID{tid}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame_bgr, f"Fatigue: {fat_pct}%", (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                frame_bgr, f"Attention: {'OK' if att_ok else 'NOT'}", (x1, y2 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        faces = len(active_fatigue)
        class_avg_fatigue = int(round(float(np.mean(active_fatigue)) if active_fatigue else 0.0))
        class_avg_attention = int(round(float(np.mean(active_attention)) if active_attention else 0.0))

        alert_active = False
        if active_fatigue and class_avg_fatigue >= config.ALERT_CLASS_AVG_PCT:
            if self.alert_start is None:
                self.alert_start = now
            elif (now - self.alert_start) >= config.ALERT_HOLD_S:
                alert_active = True
        else:
            self.alert_start = None

        cv2.putText(
            frame_bgr, f"Faces={faces}  Class Avg Fatigue={class_avg_fatigue}%",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        if alert_active:
            cv2.putText(
                frame_bgr, "ALERT: Class fatigue is high",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3
            )

        self._last_stats = EngineStats(
            faces=faces,
            class_avg_fatigue_pct=class_avg_fatigue,
            class_avg_attention_pct=class_avg_attention,
            alert_active=alert_active,
            fps=float(self._fps_smooth),
        )

        self.frame_idx += 1
        self._last_frame = frame_bgr
        return frame_bgr

    def get_frame(self):
        if not self.running or self.cap is None:
            return self._last_frame

        ok, frame = self.cap.read()
        if not ok:
            return self._last_frame

        now = time.time()
        self._update_fps(now)

        frame = cv2.flip(frame, 1)
        return self._process_frame_inplace(frame, now)

    def process_external_frame(self, frame_bgr: np.ndarray, flip: bool = False):
        if frame_bgr is None:
            return None, self.get_stats()

        now = time.time()
        self._update_fps(now)

        if flip:
            frame_bgr = cv2.flip(frame_bgr, 1)

        out = self._process_frame_inplace(frame_bgr, now)
        return out, self.get_stats()

    def get_stats(self) -> dict:
        s = self._last_stats
        return {
            "faces": s.faces,
            "class_avg_fatigue_pct": s.class_avg_fatigue_pct,
            "class_avg_attention_pct": s.class_avg_attention_pct,
            "alert_active": s.alert_active,
            "fps": s.fps,
        }