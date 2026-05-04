import os
import time
import warnings
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from openpyxl.chart import ScatterChart, Reference, Series

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import mediapipe as mp
import numpy as np

import config
from dataset_store import init_dataset_db
from db import (
    init_db,
    start_session as db_start_session,
    finish_session as db_finish_session,
    replace_session_students,
)
from face_identity import FaceIdentityManager
from head_detector import HeadDetector
from multi_face_detector import MultiFaceDetector
from tracker_flow import FlowMultiTracker
from session_stats import SessionStatsManager
from raport_manager import RaportManager
from decision_rules import evaluate_student_state

from engine_core import (
    LEFT_EYE,
    RIGHT_EYE,
    PersonState,
    EngineStats,
    StaleTrack,
    HybridYoloDetector,
    dataset_recorder_ref,
    set_dataset_recorder,
    lm_xy,
    expand_bbox,
    shrink_box,
    infer_head_box_from_face,
    eye_aspect_ratio,
    bbox_area,
    bbox_center,
    box_contains_point,
    iou,
    normalize_xyxy,
    nms_xyxy,
    nms_dict_boxes,
    looks_like_face_box,
    looks_like_head_box,
    face_inside_head_score,
    box_key,
    valid_face_area,
    valid_head_area,
    clone_person_state,
    update_boolean_alert_with_hysteresis,
    student_alert_severity,
    build_student_alert_message,
    compute_student_alerts,
    compute_class_alerts,
    maybe_trigger_sound_alert,
    export_session_report_xlsx as core_export_session_report_xlsx,
    is_box_in_reasonable_vertical_zone,
    is_face_candidate_stable,
    is_head_candidate_stable,
    mesh_confirms_face_box,
    filter_mp_faces,
    detect_yolo_faces,
    filter_yolo_faces,
    filter_heads,
    best_face_inside_head,
    compute_face_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
warnings.filterwarnings("ignore", module="google.protobuf.symbol_database")


@dataclass
class SeatMemoryItem:
    identity_id: int
    center: Tuple[float, float]
    first_seen: float
    last_seen: float
    hits: int = 1


class SeatMemoryManager:
    def __init__(self):
        self.started_at: Optional[float] = None
        self.items: Dict[int, SeatMemoryItem] = {}

    def reset(self, now: float):
        self.started_at = now
        self.items = {}

    def _center_dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update_identity(self, identity_id: int, bbox: Tuple[int, int, int, int], now: float):
        c = bbox_center(bbox)
        old = self.items.get(int(identity_id))
        if old is None:
            self.items[int(identity_id)] = SeatMemoryItem(
                identity_id=int(identity_id),
                center=c,
                first_seen=now,
                last_seen=now,
                hits=1,
            )
            return

        alpha = 0.75
        old.center = (
            alpha * old.center[0] + (1.0 - alpha) * c[0],
            alpha * old.center[1] + (1.0 - alpha) * c[1],
        )
        old.last_seen = now
        old.hits += 1

    def find_reusable_identity(
        self,
        bbox: Tuple[int, int, int, int],
        now: float,
        active_identity_ids: Set[int],
    ) -> Optional[int]:
        if not bool(getattr(config, "ENABLE_SEAT_MEMORY", True)):
            return None

        if self.started_at is None:
            return None

        if (now - self.started_at) > float(getattr(config, "SEAT_MEMORY_DURATION_S", 1.3 * 60 * 60)):
            return None

        max_missing = float(getattr(config, "SEAT_MAX_MISSING_S", 120.0))
        max_dist = float(getattr(config, "SEAT_MAX_DIST_PX", 180.0))
        cur_center = bbox_center(bbox)

        best_identity_id = None
        best_dist = 1e18

        for identity_id, item in self.items.items():
            if identity_id in active_identity_ids:
                continue

            if (now - item.last_seen) > max_missing:
                continue

            dist = self._center_dist(cur_center, item.center)
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_identity_id = identity_id

        return best_identity_id


class MonitoringEngine:
    def __init__(self):
        self._last_faces_data: List[dict] = []
        self._last_stats = EngineStats()

        self.frame_idx = 0
        self.session_active = False
        self.session_calibration_until: float = 0.0

        self.session_stats = SessionStatsManager()
        self.raport_manager = RaportManager(
            db_path=config.DB_PATH,
            recent_rapoarte_limit=int(getattr(config, "RECENT_REPORTS_IN_UI", 3)),
            interval_raport_s=int(getattr(config, "REPORT_INTERVAL_S", 60)),
        )

        self.face_detector = MultiFaceDetector(
            min_conf=config.MIN_DET_CONF if hasattr(config, "MIN_DET_CONF") else 0.4
        )
        self.yolo_detector = HybridYoloDetector()
        self.head_detector = HeadDetector()

        self.tracker = self._build_tracker()

        self.identity_manager = self._build_identity_manager()
        self.identity_frame_stride = int(getattr(config, "FACE_ID_EVERY_N_FRAMES", 6))
        if self.identity_frame_stride < 1:
            self.identity_frame_stride = 1

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=bool(getattr(config, "REFINE_LANDMARKS", True)),
            min_detection_confidence=float(getattr(config, "MIN_DET_CONF", 0.45)),
            min_tracking_confidence=float(getattr(config, "MIN_TRK_CONF", 0.45)),
        )

        self.people: Dict[int, PersonState] = {}
        self.track_identity_cache: Dict[int, Optional[int]] = {}
        self.track_persistent_id_cache: Dict[int, int] = {}
        self.next_persistent_track_id: int = 1

        self.stale_tracks: Dict[int, StaleTrack] = {}
        self.refresh_until_frame: int = -1
        self._last_periodic_refresh_frame: int = -1
        self._last_track_runtime: Dict[int, dict] = {}

        self.seat_memory = SeatMemoryManager()

        self.alert_start: Optional[float] = None
        self._last_time = time.time()
        self._fps_smooth = 0.0

        self.class_fatigue_bad_since: Optional[float] = None
        self.class_attention_bad_since: Optional[float] = None

        self.class_fatigue_alert_active: bool = False
        self.class_attention_alert_active: bool = False

        self.last_any_alert_sound_ts: float = 0.0

        init_db(config.DB_PATH)
        init_dataset_db()

        self.raport_manager.bind_session(None)

        self._cached_heads: List[dict] = []
        self._cached_yolo_faces: List[dict] = []
        self._cached_mp_faces: List[Tuple[int, int, int, int]] = []
        self._last_hybrid_detect_frame = -1

        self.face_stability = {}
        self.head_stability = {}
        self.yolo_face_stability = {}
        self.STABLE_HITS = 3
        self.STABLE_MAX_MISS = 2

    @property
    def session_started_at(self) -> Optional[float]:
        return self.session_stats.session_started_at

    @session_started_at.setter
    def session_started_at(self, value: Optional[float]) -> None:
        self.session_stats.session_started_at = value

    @property
    def session_stopped_at(self) -> Optional[float]:
        return self.session_stats.session_stopped_at

    @session_stopped_at.setter
    def session_stopped_at(self, value: Optional[float]) -> None:
        self.session_stats.session_stopped_at = value

    @property
    def session_id(self) -> Optional[int]:
        return self.session_stats.session_id

    @session_id.setter
    def session_id(self, value: Optional[int]) -> None:
        self.session_stats.session_id = value

    @property
    def session_samples(self) -> List[dict]:
        return self.session_stats.session_samples

    @session_samples.setter
    def session_samples(self, value: List[dict]) -> None:
        self.session_stats.session_samples = value

    @property
    def session_seen_track_ids(self) -> Set[int]:
        return self.session_stats.session_seen_track_ids

    @session_seen_track_ids.setter
    def session_seen_track_ids(self, value: Set[int]) -> None:
        self.session_stats.session_seen_track_ids = value

    @property
    def session_seen_identity_ids(self) -> Set[int]:
        return self.session_stats.session_seen_identity_ids

    @session_seen_identity_ids.setter
    def session_seen_identity_ids(self, value: Set[int]) -> None:
        self.session_stats.session_seen_identity_ids = value

    @property
    def session_total_face_observations(self) -> int:
        return self.session_stats.session_total_face_observations

    @session_total_face_observations.setter
    def session_total_face_observations(self, value: int) -> None:
        self.session_stats.session_total_face_observations = int(value)

    @property
    def session_valid_observations(self) -> int:
        return self.session_stats.session_valid_observations

    @session_valid_observations.setter
    def session_valid_observations(self, value: int) -> None:
        self.session_stats.session_valid_observations = int(value)

    @property
    def session_alert_event_count(self) -> int:
        return self.session_stats.session_alert_event_count

    @session_alert_event_count.setter
    def session_alert_event_count(self, value: int) -> None:
        self.session_stats.session_alert_event_count = int(value)

    @property
    def session_student_stats(self) -> Dict[str, Dict[str, Any]]:
        return self.session_stats.session_student_stats

    @session_student_stats.setter
    def session_student_stats(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.session_stats.session_student_stats = value

    @property
    def _recent_reports(self) -> List[Dict[str, Any]]:
        return self.raport_manager.recent_rapoarte

    @_recent_reports.setter
    def _recent_reports(self, value: List[Dict[str, Any]]) -> None:
        self.raport_manager.recent_rapoarte = value

    @property
    def _report_buffer(self) -> List[Dict[str, Any]]:
        return self.raport_manager.raport_buffer

    @_report_buffer.setter
    def _report_buffer(self, value: List[Dict[str, Any]]) -> None:
        self.raport_manager.raport_buffer = value

    @property
    def _last_report_ts(self) -> float:
        return self.raport_manager.last_raport_ts

    @_last_report_ts.setter
    def _last_report_ts(self, value: float) -> None:
        self.raport_manager.last_raport_ts = float(value)

    def _build_tracker(self):
        return FlowMultiTracker(
            ttl_s=config.TRACK_TTL_S,
            iou_t=config.IOU_MATCH_T,
            dist_t=config.CENTER_DIST_T,
            min_pts=config.MIN_TRACK_POINTS,
        )

    def _build_identity_manager(self):
        return FaceIdentityManager(
            sim_threshold=float(getattr(config, "FACE_ID_SIM_THRESHOLD", 0.75)),
            min_face_size=int(getattr(config, "FACE_ID_MIN_FACE_SIZE", 90)),
            min_blur_score=float(getattr(config, "FACE_ID_MIN_BLUR_SCORE", 100.0)),
            candidate_hits=int(getattr(config, "FACE_ID_CANDIDATE_HITS", 6)),
            candidate_ttl_s=float(getattr(config, "FACE_ID_CANDIDATE_TTL_S", 3.0)),
        )

    def _cleanup_stale_refresh_cache(self, now: float):
        ttl_s = float(getattr(config, "REFRESH_CACHE_TTL_S", 2.0))
        to_del = []
        for key, item in self.stale_tracks.items():
            if (now - item.saved_at) > ttl_s:
                to_del.append(key)

        for key in to_del:
            self.stale_tracks.pop(key, None)

    def _center_dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ensure_persistent_id(self, raw_tid: int) -> int:
        pid = self.track_persistent_id_cache.get(int(raw_tid))
        if pid is not None:
            return int(pid)

        pid = int(self.next_persistent_track_id)
        self.next_persistent_track_id += 1
        self.track_persistent_id_cache[int(raw_tid)] = pid
        return pid

    def _snapshot_runtime_for_refresh(self, now: float):
        self.stale_tracks = {}
        for raw_tid, item in self._last_track_runtime.items():
            bbox = item.get("bbox")
            if bbox is None:
                continue

            person_state = self.people.get(int(raw_tid), PersonState())
            stale = StaleTrack(
                raw_track_id=int(raw_tid),
                persistent_id=int(item.get("persistent_id", self._ensure_persistent_id(int(raw_tid)))),
                identity_id=item.get("identity_id"),
                bbox=tuple(map(int, bbox)),
                center=bbox_center(bbox),
                saved_at=now,
                person_state=clone_person_state(person_state),
            )
            self.stale_tracks[int(raw_tid)] = stale

    def _periodic_refresh_runtime(self, now: float):
        self._snapshot_runtime_for_refresh(now)

        self.tracker = self._build_tracker()
        self.people = {}
        self.track_identity_cache = {}
        self.track_persistent_id_cache = {}
        self._cached_heads = []
        self._cached_yolo_faces = []
        self._cached_mp_faces = []
        self._last_hybrid_detect_frame = -1

        self.face_stability = {}
        self.head_stability = {}
        self.yolo_face_stability = {}
        self.STABLE_HITS = 3
        self.STABLE_MAX_MISS = 2

        self.refresh_until_frame = self.frame_idx + int(getattr(config, "REFRESH_GRACE_FRAMES", 15))
        self._last_periodic_refresh_frame = self.frame_idx

    def _maybe_run_periodic_refresh(self, now: float):
        if not bool(getattr(config, "ENABLE_PERIODIC_REFRESH", False)):
            return

        refresh_every = int(getattr(config, "REFRESH_EVERY_N_FRAMES", 240))
        if refresh_every < 1:
            return

        if self.frame_idx <= 0:
            return

        if self.frame_idx == self._last_periodic_refresh_frame:
            return

        if self.frame_idx % refresh_every != 0:
            return

        self._periodic_refresh_runtime(now)

    def _try_reassociate_after_refresh(self, current_tracks: Dict[int, Tuple[int, int, int, int]], now: float):
        if self.frame_idx > self.refresh_until_frame:
            self._cleanup_stale_refresh_cache(now)
            return

        self._cleanup_stale_refresh_cache(now)
        if not self.stale_tracks:
            return

        max_center_dist = float(getattr(config, "REFRESH_REID_CENTER_DIST_T", 110))
        min_iou = float(getattr(config, "REFRESH_REID_IOU_T", 0.18))

        used_stale_keys: Set[int] = set()

        for raw_tid, box in current_tracks.items():
            raw_tid = int(raw_tid)

            if raw_tid in self.track_persistent_id_cache:
                continue

            best_key = None
            best_score = -1e18
            cur_center = bbox_center(box)

            for stale_key, stale in self.stale_tracks.items():
                if stale_key in used_stale_keys:
                    continue

                overlap = iou(box, stale.bbox)
                dist = self._center_dist(cur_center, stale.center)

                if overlap < min_iou and dist > max_center_dist:
                    continue

                score = overlap * 1000.0 - dist

                if score > best_score:
                    best_score = score
                    best_key = stale_key

            if best_key is None:
                continue

            stale = self.stale_tracks.get(best_key)
            if stale is None:
                continue

            used_stale_keys.add(best_key)

            self.track_persistent_id_cache[raw_tid] = int(stale.persistent_id)
            self.track_identity_cache[raw_tid] = stale.identity_id
            self.people[raw_tid] = clone_person_state(stale.person_state)

        for stale_key in used_stale_keys:
            self.stale_tracks.pop(stale_key, None)

    def _reset_live_runtime_state(self):
        self._last_faces_data = []
        self._last_stats = EngineStats()

        self.people = {}
        self.frame_idx = 0
        self.alert_start = None

        self.tracker = self._build_tracker()

        self.identity_manager = self._build_identity_manager()
        self.track_identity_cache = {}
        self.track_persistent_id_cache = {}
        self.next_persistent_track_id = 1

        self.stale_tracks = {}
        self.refresh_until_frame = -1
        self._last_periodic_refresh_frame = -1
        self._last_track_runtime = {}

        self.seat_memory = SeatMemoryManager()

        self._last_time = time.time()
        self._fps_smooth = 0.0

        self._cached_heads = []
        self._cached_yolo_faces = []
        self._cached_mp_faces = []
        self._last_hybrid_detect_frame = -1

        self.face_stability = {}
        self.head_stability = {}
        self.yolo_face_stability = {}
        self.STABLE_HITS = 3
        self.STABLE_MAX_MISS = 2

        self.class_fatigue_bad_since = None
        self.class_attention_bad_since = None
        self.class_fatigue_alert_active = False
        self.class_attention_alert_active = False
        self.last_any_alert_sound_ts = 0.0

        self._report_buffer = []
        self._last_report_ts = time.time()

    def _reset_session_state(self):
        self.session_stats.reset()

    def _update_session_student_stats(
        self,
        student_key: str,
        fatigue_pct: float,
        attention_pct: float,
        alert_type: str = "none",
        severity: str = "none",
        reason: str = "",
        decision: str = "",
    ) -> None:
        self.session_stats.update_student_stats(
            student_key=student_key,
            fatigue_pct=fatigue_pct,
            attention_pct=attention_pct,
            alert_type=alert_type,
            severity=severity,
            reason=reason,
            decision=decision,
        )

    def _build_session_students_summary(self) -> List[Dict[str, Any]]:
        return self.session_stats.build_students_summary()

    def _class_alert_type_and_level(
        self,
        fatigue_alert_active: bool,
        attention_alert_active: bool,
        active_students: int,
    ) -> Tuple[str, str]:
        if active_students <= 0:
            return "no_students_detected", "warning"

        if fatigue_alert_active and attention_alert_active:
            return "fatigue_attention", "critical"
        if fatigue_alert_active:
            return "fatigue", "warning"
        if attention_alert_active:
            return "attention", "warning"
        return "none", "none"

    def _class_reason_and_decision(
        self,
        class_avg_fatigue: float,
        class_avg_attention: float,
        fatigue_alert_active: bool,
        attention_alert_active: bool,
        active_students: int,
        class_decision_explanation: str,
    ) -> Tuple[str, str]:
        if active_students <= 0:
            return (
                "No active students detected in the monitored scene.",
                "Check camera position, focus, visibility, and classroom framing.",
            )

        if fatigue_alert_active and attention_alert_active:
            return (
                f"Class fatigue average is {round(class_avg_fatigue, 2)} and class attention average is {round(class_avg_attention, 2)}; both crossed alert thresholds.",
                "Recommend a short break and re-engage the class with direct questions.",
            )
        if fatigue_alert_active:
            return (
                f"Class fatigue average reached {round(class_avg_fatigue, 2)} and exceeded the configured fatigue threshold.",
                "Recommend a short break or a lighter activity.",
            )
        if attention_alert_active:
            return (
                f"Class attention average dropped to {round(class_avg_attention, 2)} and fell below the configured attention threshold.",
                "Re-engage the class with direct questions or change activity pace.",
            )

        return (
            class_decision_explanation or "No class-level alert is active.",
            "No intervention needed. Continue monitoring.",
        )

    def start_session(self):
        self.session_active = True
        self._reset_session_state()
        self._reset_live_runtime_state()

        started_at = time.time()
        self.session_calibration_until = started_at + float(getattr(config, "SESSION_CALIBRATION_SECONDS", 6.0))
        self.session_started_at = started_at
        self.session_stopped_at = None
        self.seat_memory.reset(started_at)

        self.session_id = db_start_session(
            config.DB_PATH,
            source="webcam",
            notes="Session started from frontend",
        )

        self.session_stats.start(session_id=self.session_id, started_at=started_at)
        self.raport_manager.bind_session(self.session_id)

    def stop_session(self):
        self.session_active = False
        self.session_stopped_at = time.time()
        self.session_stats.stop(self.session_stopped_at)

        if self.session_id is None:
            return

        summary = self.get_session_summary()

        class_alert_type, class_alert_level = self._class_alert_type_and_level(
            fatigue_alert_active=bool(self.class_fatigue_alert_active),
            attention_alert_active=bool(self.class_attention_alert_active),
            active_students=int(round(summary.get("avg_faces", 0.0))),
        )

        reason, decision = self._class_reason_and_decision(
            class_avg_fatigue=float(summary.get("avg_fatigue", 0.0)),
            class_avg_attention=float(summary.get("avg_attention", 0.0)),
            fatigue_alert_active=bool(self.class_fatigue_alert_active),
            attention_alert_active=bool(self.class_attention_alert_active),
            active_students=int(round(summary.get("avg_faces", 0.0))),
            class_decision_explanation=self._last_stats.decision_explanation if hasattr(self._last_stats, "decision_explanation") else "",
        )

        db_finish_session(
            db_path=config.DB_PATH,
            session_id=self.session_id,
            faces_avg=float(summary.get("avg_faces", 0.0)),
            heads_avg=float(summary.get("avg_heads", 0.0)),
            active_students_avg=float(summary.get("avg_faces", 0.0)),
            fatigue_avg=float(summary.get("avg_fatigue", 0.0)),
            attention_avg=float(summary.get("avg_attention", 0.0)),
            class_alert_type=class_alert_type,
            class_alert_level=class_alert_level,
            reason=reason,
            decision=decision,
            report_path="",
            status="finished",
        )

        session_students = self._build_session_students_summary()
        replace_session_students(
            config.DB_PATH,
            self.session_id,
            session_students,
        )

        self._recent_reports = self.raport_manager.recent_rapoarte

    def _append_session_sample(
        self,
        now: float,
        faces: int,
        heads: int,
        fatigue: int,
        attention: int,
        alert_active: bool,
    ):
        if not self.session_active:
            return

        self.session_stats.append_sample(
            now=now,
            faces=faces,
            heads=heads,
            fatigue=fatigue,
            attention=attention,
            alert_active=alert_active,
        )

    def _is_session_calibrating(self, now: float) -> bool:
        return bool(self.session_active and now < float(getattr(self, "session_calibration_until", 0.0)))

    def has_session_data(self) -> bool:
        return self.session_stats.has_data()

    def get_session_summary(self) -> dict:
        return self.session_stats.get_summary()

    def export_session_raport_xlsx(self) -> str:
        session_students = self._build_session_students_summary()
        return self.raport_manager.exporta_raport_sesiune_xlsx(
            session_id=self.session_id,
            session_summary=self.get_session_summary(),
            session_students=session_students,
        )

    def export_session_report_xlsx(self) -> str:
        return self.export_session_raport_xlsx()

    def _update_fps(self, now: float):
        dt = now - self._last_time
        self._last_time = now
        fps_inst = (1.0 / dt) if dt > 1e-6 else 0.0
        self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps_inst

    def _maybe_store_report(
        self,
        now: float,
        faces: int,
        heads: int,
        fatigue: int,
        attention: int,
        fatigue_alert_active: bool,
        attention_alert_active: bool,
        student_alerts: List[dict],
        studenti_runtime: List[dict],
        class_decision_explanation: str,
    ):
        self.raport_manager.adauga_esantion_runtime(
            now=now,
            faces=faces,
            heads=heads,
            fatigue=fatigue,
            attention=attention,
            fatigue_alert_active=fatigue_alert_active,
            attention_alert_active=attention_alert_active,
            student_alerts=student_alerts,
            studenti_runtime=studenti_runtime,
            class_decision_explanation=class_decision_explanation,
        )

        if not self.session_active:
            return

        self.raport_manager.poate_stoca_snapshot(
            session_id=self.session_id,
            now=now,
            class_alert_type_and_level_fn=self._class_alert_type_and_level,
            class_reason_and_decision_fn=self._class_reason_and_decision,
        )
        self._recent_reports = self.raport_manager.recent_rapoarte

    def _save_dataset_sample(
        self,
        frame_bgr: np.ndarray,
        visible_face_track_ids_this_frame: List[int],
        head_count_this_frame: int,
        all_boxes_for_dataset: List[dict],
    ):
        import engine_core

        if engine_core.dataset_recorder_ref is None:
            return

        extra_payload = {
            "tracked_face_ids": [int(x) for x in visible_face_track_ids_this_frame],
            "head_count": int(head_count_this_frame),
            "boxes": [
                {
                    "id": int(item["id"]),
                    "kind": item.get("kind", "face"),
                    "bbox_px": [
                        int(item["bbox_px"][0]),
                        int(item["bbox_px"][1]),
                        int(item["bbox_px"][2]),
                        int(item["bbox_px"][3]),
                    ],
                    "inferred": bool(item.get("inferred", False)),
                }
                for item in all_boxes_for_dataset
            ],
        }

        engine_core.dataset_recorder_ref.maybe_save(
            frame=frame_bgr.copy(),
            extra=extra_payload,
        )

    def _match_track_to_head(self, track_box, head_candidates):
        best_idx = -1
        best_score = -1.0
        tcx, tcy = bbox_center(track_box)

        for idx, item in enumerate(head_candidates):
            head_box = item["bbox"]
            overlap = iou(track_box, head_box)
            hc_x, hc_y = bbox_center(head_box)
            dist = ((tcx - hc_x) ** 2 + (tcy - hc_y) ** 2) ** 0.5
            score = overlap * 1000.0 - dist
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _stability_iou(self, box_a, box_b) -> float:
        try:
            return float(iou(box_a, box_b))
        except Exception:
            return 0.0

    def _update_candidate_stability(
        self,
        state: Dict[Tuple[int, int, int, int], dict],
        boxes: List[Tuple[int, int, int, int]],
        *,
        hits_needed: int,
        max_miss: int,
        match_iou: float,
    ) -> List[Tuple[int, int, int, int]]:
        matched_prev: Set[Tuple[int, int, int, int]] = set()
        stable: List[Tuple[int, int, int, int]] = []
        next_state: Dict[Tuple[int, int, int, int], dict] = {}

        for box in boxes:
            best_key = None
            best_iou = 0.0
            for prev_key, entry in state.items():
                cur_iou = self._stability_iou(box, entry.get("bbox", prev_key))
                if cur_iou >= match_iou and cur_iou > best_iou and prev_key not in matched_prev:
                    best_iou = cur_iou
                    best_key = prev_key

            if best_key is None:
                entry = {"bbox": box, "hits": 1, "miss": 0}
            else:
                prev = state[best_key]
                matched_prev.add(best_key)
                px1, py1, px2, py2 = prev.get("bbox", best_key)
                bx1, by1, bx2, by2 = box
                alpha = 0.65
                smoothed = (
                    int(px1 * alpha + bx1 * (1.0 - alpha)),
                    int(py1 * alpha + by1 * (1.0 - alpha)),
                    int(px2 * alpha + bx2 * (1.0 - alpha)),
                    int(py2 * alpha + by2 * (1.0 - alpha)),
                )
                entry = {"bbox": smoothed, "hits": int(prev.get("hits", 0)) + 1, "miss": 0}

            next_state[tuple(entry["bbox"])] = entry
            if int(entry["hits"]) >= hits_needed:
                stable.append(tuple(entry["bbox"]))

        for prev_key, prev in state.items():
            if prev_key in matched_prev:
                continue
            miss = int(prev.get("miss", 0)) + 1
            if miss <= max_miss:
                carry = {"bbox": tuple(prev.get("bbox", prev_key)), "hits": int(prev.get("hits", 0)), "miss": miss}
                next_state[tuple(carry["bbox"])] = carry
                if int(carry["hits"]) >= hits_needed:
                    stable.append(tuple(carry["bbox"]))

        state.clear()
        state.update(next_state)
        return nms_xyxy(stable, iou_thresh=0.45)

    def _stabilize_attention_score(self, tid: int, raw_attention: float, now: float) -> float:
        st = self.people.get(int(tid))
        if st is None:
            return float(raw_attention)

        low_threshold = float(getattr(config, "ATTENTION_LOW_THRESHOLD", 50.0))
        recovery_threshold = float(getattr(config, "ATTENTION_RECOVERY_THRESHOLD", 60.0))
        confirm_seconds = float(getattr(config, "ATTENTION_LOW_CONFIRM_SECONDS", 120.0))
        short_drop_value = float(getattr(config, "ATTENTION_SHORT_DROP_VALUE", 85.0))

        raw_attention = float(raw_attention)

        if not hasattr(st, "attention_low_since"):
            st.attention_low_since = None
        if not hasattr(st, "last_stable_attention"):
            st.last_stable_attention = 100.0

        if raw_attention < low_threshold:
            if st.attention_low_since is None:
                st.attention_low_since = now

            low_duration = now - float(st.attention_low_since)

            if low_duration < confirm_seconds:
                stabilized = max(short_drop_value, float(st.last_stable_attention) * 0.90)
            else:
                stabilized = raw_attention
        else:
            if raw_attention >= recovery_threshold:
                st.attention_low_since = None
            stabilized = raw_attention

        st.last_stable_attention = float(stabilized)
        return float(stabilized)

    def _tight_head_from_face(self, face_box, w_img, h_img):
        x1, y1, x2, y2 = face_box
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        cx = (x1 + x2) * 0.5
        face_bottom = y2

        head_w = fw * float(getattr(config, "DISPLAY_HEAD_FROM_FACE_W_SCALE", 1.32))
        head_h = fh * float(getattr(config, "DISPLAY_HEAD_FROM_FACE_H_SCALE", 1.58))

        nx1 = int(max(0, cx - head_w * 0.5))
        nx2 = int(min(w_img - 1, cx + head_w * 0.5))
        ny2 = int(min(h_img - 1, face_bottom))
        ny1 = int(max(0, ny2 - head_h))

        out = (nx1, ny1, nx2, ny2)
        return out if looks_like_head_box(out, w_img, h_img) else infer_head_box_from_face(face_box, w_img, h_img)

    def _process_frame_inplace(self, frame_bgr: np.ndarray, now: float, want_boxes: bool) -> None:
        self._maybe_run_periodic_refresh(now)

        self.tracker.update_optical_flow(frame_bgr, now)

        detect_every = int(getattr(config, "DETECT_EVERY_N_FRAMES", 3))
        if detect_every < 1:
            detect_every = 1

        if self.frame_idx % detect_every == 0:
            dets = filter_mp_faces(self.face_detector, frame_bgr)
            self.tracker.sync_with_detections(frame_bgr, dets, now)

        self.tracker.drop_stale(now)
        tracks = self.tracker.get_tracks(frame_bgr)

        self._try_reassociate_after_refresh(tracks, now)

        for tid in tracks.keys():
            tid = int(tid)
            if tid not in self.people:
                self.people[tid] = PersonState()
            self._ensure_persistent_id(tid)

        active_fatigue: List[int] = []
        active_attention: List[float] = []

        confirmed_track_ids_this_frame: List[int] = []
        visible_face_track_ids_this_frame: List[int] = []

        h_img, w_img = frame_bgr.shape[:2]
        face_mesh_scale = float(getattr(config, "FACE_MESH_EXPAND_SCALE", 1.18))

        hybrid_stride = int(getattr(config, "HEAD_DETECT_EVERY_N_FRAMES", 2))
        if hybrid_stride < 1:
            hybrid_stride = 1

        if (
            self._last_hybrid_detect_frame < 0
            or (self.frame_idx - self._last_hybrid_detect_frame) >= hybrid_stride
        ):
            self._cached_heads = filter_heads(self.yolo_detector, self.head_detector, frame_bgr)
            self._cached_yolo_faces = filter_yolo_faces(self.yolo_detector, frame_bgr)
            self._cached_mp_faces = filter_mp_faces(self.face_detector, frame_bgr)
            self._last_hybrid_detect_frame = self.frame_idx

        cached_heads = list(self._cached_heads)
        cached_yolo_faces = list(self._cached_yolo_faces)
        cached_mp_faces = list(self._cached_mp_faces)

        cached_mp_faces = self._update_candidate_stability(
            self.face_stability,
            list(cached_mp_faces),
            hits_needed=int(getattr(config, "FACE_STABLE_HITS", self.STABLE_HITS)),
            max_miss=int(getattr(config, "FACE_STABLE_MAX_MISS", self.STABLE_MAX_MISS)),
            match_iou=float(getattr(config, "FACE_STABLE_MATCH_IOU", 0.35)),
        )

        stable_yolo_faces = self._update_candidate_stability(
            getattr(self, "yolo_face_stability", {}),
            [tuple(x["bbox"]) for x in cached_yolo_faces],
            hits_needed=int(getattr(config, "YOLO_FACE_STABLE_HITS", 2)),
            max_miss=int(getattr(config, "YOLO_FACE_STABLE_MAX_MISS", 2)),
            match_iou=float(getattr(config, "YOLO_FACE_STABLE_MATCH_IOU", 0.30)),
        )
        cached_yolo_faces = [
            {"kind": "face", "bbox": tuple(b), "conf": 0.5}
            for b in stable_yolo_faces
        ]

        stable_heads = self._update_candidate_stability(
            self.head_stability,
            [tuple(x["bbox"]) for x in cached_heads],
            hits_needed=int(getattr(config, "HEAD_STABLE_HITS", 2)),
            max_miss=int(getattr(config, "HEAD_STABLE_MAX_MISS", 2)),
            match_iou=float(getattr(config, "HEAD_STABLE_MATCH_IOU", 0.30)),
        )
        cached_heads = [
            {"kind": "head", "bbox": tuple(b), "conf": 0.5}
            for b in stable_heads
        ]

        used_head_idxs: Set[int] = set()
        display_pairs: List[dict] = []
        all_boxes_for_dataset: List[dict] = []
        next_runtime_snapshot: Dict[int, dict] = {}
        validated_head_boxes: List[Tuple[int, int, int, int]] = []

        for tid, (x1, y1, x2, y2) in tracks.items():
            tid = int(tid)

            x1 = max(0, min(w_img - 1, int(x1)))
            y1 = max(0, min(h_img - 1, int(y1)))
            x2 = max(0, min(w_img - 1, int(x2)))
            y2 = max(0, min(h_img - 1, int(y2)))

            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            raw_box = (x1, y1, x2, y2)
            if not is_face_candidate_stable(raw_box, w_img, h_img):
                continue

            persistent_id = self._ensure_persistent_id(tid)

            head_idx = self._match_track_to_head(raw_box, cached_heads) if cached_heads else -1
            matched_head = None
            head_inferred = False

            if head_idx >= 0:
                matched_head = cached_heads[head_idx]["bbox"]
                if iou(raw_box, matched_head) > 0.02:
                    used_head_idxs.add(head_idx)
                else:
                    matched_head = None

            face_box = raw_box
            face_kind = "face"

            if matched_head is not None:
                candidate_face, candidate_kind, _source = best_face_inside_head(
                    face_detector=self.face_detector,
                    yolo_faces=cached_yolo_faces,
                    frame_bgr=frame_bgr,
                    head_box=matched_head,
                    mp_faces=cached_mp_faces,
                )
                if candidate_face is not None:
                    face_box = candidate_face
                    face_kind = candidate_kind or "face"

            if not is_face_candidate_stable(face_box, w_img, h_img):
                continue

            if matched_head is None:
                inferred_head = self._tight_head_from_face(face_box, w_img, h_img)
                if inferred_head is not None:
                    matched_head = inferred_head
                    head_inferred = True

            identity_crop_box = expand_bbox(face_box, w_img, h_img, scale=1.06)
            if identity_crop_box is None:
                identity_crop_box = face_box

            ix1, iy1, ix2, iy2 = identity_crop_box
            identity_face = frame_bgr[iy1:iy2, ix1:ix2]
            if identity_face.size == 0:
                continue

            face_h, face_w = identity_face.shape[:2]
            if face_w < getattr(config, "FACE_BOX_MIN_SIZE", 16) or face_h < getattr(config, "FACE_BOX_MIN_SIZE", 16):
                continue

            mesh_ok, lms, mesh_w, mesh_h = mesh_confirms_face_box(
                mesh=self.mesh,
                frame_bgr=frame_bgr,
                face_box=face_box,
                scale=face_mesh_scale,
            )
            if not mesh_ok:
                continue

            visible_face_track_ids_this_frame.append(int(persistent_id))

            identity_id = self.track_identity_cache.get(tid)
            if self.frame_idx % self.identity_frame_stride == 0:
                new_identity_id = self.identity_manager.update_track(tid, identity_face, now)
                if new_identity_id is not None:
                    identity_id = int(new_identity_id)
                    self.track_identity_cache[tid] = identity_id
                elif bool(getattr(config, "ENABLE_SEAT_MEMORY", True)):
                    active_identity_ids = {
                        int(v) for v in self.track_identity_cache.values() if v is not None
                    }

                    reused_identity_id = self.seat_memory.find_reusable_identity(
                        bbox=face_box,
                        now=now,
                        active_identity_ids=active_identity_ids,
                    )

                    if reused_identity_id is not None:
                        identity_id = int(reused_identity_id)
                        self.track_identity_cache[tid] = identity_id

            if identity_id is not None and self.session_active:
                st_for_identity = self.people.get(tid)
                if st_for_identity is not None and getattr(st_for_identity, "valid_observations", 0) >= int(getattr(config, "IDENTITY_MIN_VALID_OBSERVATIONS", 10)):
                    self.session_seen_identity_ids.add(int(identity_id))

            if identity_id is not None and bool(getattr(config, "ENABLE_SEAT_MEMORY", True)):
                self.seat_memory.update_identity(
                    identity_id=int(identity_id),
                    bbox=face_box,
                    now=now,
                )

            st = self.people[tid]
            try:
                fat_pct, att_score = compute_face_metrics(
                    lms=lms,
                    mesh_w=mesh_w,
                    mesh_h=mesh_h,
                    now=now,
                    st=st,
                )
            except Exception:
                continue

            att_score = self._stabilize_attention_score(tid, att_score, now)
            st.attention_pct = float(att_score)
            st.fatigue_pct = float(fat_pct)

            active_fatigue.append(int(fat_pct))
            active_attention.append(float(att_score))
            self.session_valid_observations += 1

            confirmed_track_ids_this_frame.append(int(tid))

            display_id = int(identity_id) if identity_id is not None else int(persistent_id)

            display_pairs.append(
                {
                    "id": display_id,
                    "persistent_id": int(persistent_id),
                    "raw_tid": int(tid),
                    "kind": "face",
                    "face_kind": face_kind,
                    "face_box": face_box,
                    "head_box": matched_head,
                    "head_inferred": bool(head_inferred),
                    "identity_id": identity_id,
                    "valid_observations": int(st.valid_observations),
                }
            )

            next_runtime_snapshot[int(tid)] = {
                "bbox": face_box,
                "persistent_id": int(persistent_id),
                "identity_id": identity_id,
            }

            all_boxes_for_dataset.append(
                {
                    "id": display_id,
                    "kind": face_kind,
                    "bbox_px": face_box,
                    "inferred": False,
                }
            )

            if matched_head is not None:
                validated_head_boxes.append(matched_head)
                all_boxes_for_dataset.append(
                    {
                        "id": display_id,
                        "kind": "head",
                        "bbox_px": matched_head,
                        "inferred": bool(head_inferred),
                    }
                )

        head_only_pairs = []
        for idx, item in enumerate(cached_heads):
            if idx in used_head_idxs:
                continue

            self_face_box, self_face_kind, _source = best_face_inside_head(
                face_detector=self.face_detector,
                yolo_faces=cached_yolo_faces,
                frame_bgr=frame_bgr,
                head_box=item["bbox"],
                mp_faces=cached_mp_faces,
            )

            if self_face_box is None:
                continue

            mesh_ok, _lms, _mw, _mh = mesh_confirms_face_box(
                mesh=self.mesh,
                frame_bgr=frame_bgr,
                face_box=self_face_box,
                scale=face_mesh_scale,
            )
            if not mesh_ok:
                continue

            head_only_id = 10000 + idx
            validated_head_boxes.append(item["bbox"])

            head_only_pairs.append(
                {
                    "id": head_only_id,
                    "kind": "face",
                    "face_kind": self_face_kind or "face",
                    "face_box": self_face_box,
                    "head_box": item["bbox"],
                    "head_inferred": False,
                    "valid_observations": 0,
                }
            )

            all_boxes_for_dataset.append(
                {
                    "id": head_only_id,
                    "kind": self_face_kind or "face",
                    "bbox_px": self_face_box,
                    "inferred": False,
                }
            )
            all_boxes_for_dataset.append(
                {
                    "id": head_only_id,
                    "kind": "head",
                    "bbox_px": item["bbox"],
                    "inferred": False,
                }
            )

        display_pairs.extend(head_only_pairs)

        validated_head_boxes = nms_xyxy(validated_head_boxes, iou_thresh=0.35)
        face_count = len(display_pairs)
        head_count = len(validated_head_boxes)
        if face_count > 0 and head_count < face_count:
            head_count = face_count

        if self.session_active and visible_face_track_ids_this_frame:
            self.session_seen_track_ids.update(int(x) for x in visible_face_track_ids_this_frame)
            self.session_total_face_observations += len(visible_face_track_ids_this_frame)

        self.identity_manager.cleanup(now, confirmed_track_ids_this_frame)

        for tid in list(self.track_identity_cache.keys()):
            if tid not in tracks:
                del self.track_identity_cache[tid]

        for tid in list(self.track_persistent_id_cache.keys()):
            if tid not in tracks:
                del self.track_persistent_id_cache[tid]

        for tid in list(self.people.keys()):
            if tid not in tracks:
                del self.people[tid]

        if want_boxes:
            payload = []
            for item in display_pairs:
                row = {
                    "id": int(item["id"]),
                    "kind": "face",
                    "face_kind": item.get("face_kind", "face"),
                    "face_bbox_n": normalize_xyxy(item["face_box"], w_img, h_img),
                    "bbox_n": normalize_xyxy(item["face_box"], w_img, h_img),
                    "head_inferred": bool(item.get("head_inferred", False)),
                    "valid_observations": int(item.get("valid_observations", 0)),
                }
                if item["head_box"] is not None:
                    row["head_bbox_n"] = normalize_xyxy(item["head_box"], w_img, h_img)
                payload.append(row)
            self._last_faces_data = payload
        else:
            self._last_faces_data = []

        faces_for_stats = face_count
        if faces_for_stats == 0 and getattr(config, "COUNT_HEADS_WHEN_NO_FACE", True):
            faces_for_stats = head_count

        class_avg_fatigue = int(round(float(np.mean(active_fatigue)) if active_fatigue else 0.0))
        class_avg_attention = int(round(float(np.mean(active_attention)) if active_attention else 0.0))

        student_alerts, new_student_alert_event, new_student_alert_kind = compute_student_alerts(
            self.people, now
        )

        (
            class_alert_active,
            fatigue_alert_active,
            attention_alert_active,
            new_class_alert_event,
            new_class_alert_kind,
            class_decision_explanation,
            self.class_fatigue_bad_since,
            self.class_attention_bad_since,
        ) = compute_class_alerts(
            class_avg_fatigue=class_avg_fatigue,
            class_avg_attention=class_avg_attention,
            active_student_count=len(active_attention),
            now=now,
            class_fatigue_alert_active=self.class_fatigue_alert_active,
            class_attention_alert_active=self.class_attention_alert_active,
            class_fatigue_bad_since=self.class_fatigue_bad_since,
            class_attention_bad_since=self.class_attention_bad_since,
        )

        self.class_fatigue_alert_active = bool(fatigue_alert_active)
        self.class_attention_alert_active = bool(attention_alert_active)

        allow_individual_alerts = bool(getattr(config, "ENABLE_INDIVIDUAL_ALERTS", True))
        has_student_alerts = bool(student_alerts) if allow_individual_alerts else False

        alert_active = bool(class_alert_active or has_student_alerts)
        new_alert_event = bool(new_class_alert_event or (allow_individual_alerts and new_student_alert_event))
        new_alert_kind = new_class_alert_kind or (new_student_alert_kind if allow_individual_alerts else "")

        is_calibrating = self._is_session_calibrating(now)
        stored_new_alert_event = bool(new_alert_event and not is_calibrating)

        if stored_new_alert_event:
            self.session_alert_event_count += 1

        sound_alert_tick, self.last_any_alert_sound_ts = maybe_trigger_sound_alert(
            now=now,
            should_trigger=stored_new_alert_event,
            last_any_alert_sound_ts=self.last_any_alert_sound_ts,
        )

        decision_parts = [class_decision_explanation]
        if allow_individual_alerts and student_alerts:
            top = student_alerts[0]
            decision_parts.append(
                f"top student alert: student {top['student_id']} fatigue={top['fatigue_pct']}% attention={top['attention_pct']}% severity={top['severity']} obs={top['valid_observations']}"
            )
        elif allow_individual_alerts:
            decision_parts.append("no individual student alert")

        decision_explanation = " | ".join(x for x in decision_parts if x)

        if not is_calibrating:
            for item in student_alerts or []:
                student_key = str(item.get("student_id"))
                fatigue_pct = float(item.get("fatigue_pct", 0.0))
                attention_pct = float(item.get("attention_pct", 0.0))
                decision_data = evaluate_student_state(attention_pct, fatigue_pct)
                severity = str(decision_data.get("severity", item.get("severity", "none")) or "none")
                reason = str(item.get("message", "") or decision_data.get("raport_message", ""))
                decision = str(decision_data.get("ui_message", "No action needed."))
                alert_type = str(decision_data.get("alert_type", "none") or "none")

                self._update_session_student_stats(
                    student_key=student_key,
                    fatigue_pct=fatigue_pct,
                    attention_pct=attention_pct,
                    alert_type=alert_type,
                    severity=severity,
                    reason=reason,
                    decision=decision,
                )

            if not student_alerts and display_pairs:
                for item in display_pairs:
                    student_key = str(item["id"])
                    baseline_attention = float(class_avg_attention)
                    baseline_fatigue = float(class_avg_fatigue)
                    decision_data = evaluate_student_state(baseline_attention, baseline_fatigue)
                    self._update_session_student_stats(
                        student_key=student_key,
                        fatigue_pct=baseline_fatigue,
                        attention_pct=baseline_attention,
                        alert_type="none",
                        severity="none",
                        reason="No individual alert was triggered for this student in the current observation.",
                        decision=str(decision_data.get("ui_message", "No action needed.")),
                    )

            self._append_session_sample(
                now=now,
                faces=faces_for_stats,
                heads=head_count,
                fatigue=class_avg_fatigue,
                attention=class_avg_attention,
                alert_active=alert_active,
            )

        studenti_runtime = []
        for item in display_pairs:
            student_id = str(item["id"])
            st_runtime = self.people.get(int(item.get("raw_tid", -1)))
            fatigue_pct = float(getattr(st_runtime, "fatigue_pct", class_avg_fatigue) if st_runtime is not None else class_avg_fatigue)
            attention_pct = float(getattr(st_runtime, "attention_pct", class_avg_attention) if st_runtime is not None else class_avg_attention)
            studenti_runtime.append({
                "student_id": student_id,
                "fatigue_pct": fatigue_pct,
                "attention_pct": attention_pct,
            })

        if not is_calibrating:
            self._maybe_store_report(
                now=now,
                faces=faces_for_stats,
                heads=head_count,
                fatigue=class_avg_fatigue,
                attention=class_avg_attention,
                fatigue_alert_active=bool(fatigue_alert_active),
                attention_alert_active=bool(attention_alert_active),
                student_alerts=student_alerts if allow_individual_alerts else [],
                studenti_runtime=studenti_runtime,
                class_decision_explanation=class_decision_explanation,
            )

            self._save_dataset_sample(
                frame_bgr=frame_bgr,
                visible_face_track_ids_this_frame=visible_face_track_ids_this_frame,
                head_count_this_frame=head_count,
                all_boxes_for_dataset=all_boxes_for_dataset,
            )

        total_detections = int(len(display_pairs))
        mesh_confirmations = int(len(confirmed_track_ids_this_frame))
        stable_tracks = int(len(visible_face_track_ids_this_frame))
        consistent_head_face = int(min(face_count, head_count))

        min_quality_samples = int(getattr(config, "LIVE_QUALITY_MIN_SAMPLES", 20))
        if min_quality_samples < 1:
            min_quality_samples = 1

        if total_detections < min_quality_samples:
            sample_factor = total_detections / max(min_quality_samples, 1)
        else:
            sample_factor = 1.0

        mesh_confirmation_rate = round(
            100.0 * mesh_confirmations / max(total_detections, 1) * sample_factor,
            2,
        )
        stable_track_rate = round(
            100.0 * stable_tracks / max(total_detections, 1) * sample_factor,
            2,
        )
        head_face_consistency_pct = round(
            100.0 * consistent_head_face / max(max(face_count, head_count), 1) * sample_factor,
            2,
        )

        raw_live_quality_pct = (
            0.4 * mesh_confirmation_rate
            + 0.3 * stable_track_rate
            + 0.3 * head_face_consistency_pct
        )
        live_quality_pct = round(min(raw_live_quality_pct, 98.0), 2)

        self._last_stats = EngineStats(
            faces=faces_for_stats,
            heads=head_count,
            class_avg_fatigue_pct=class_avg_fatigue,
            class_avg_attention_pct=class_avg_attention,
            live_quality_pct=live_quality_pct,
            mesh_confirmation_rate=mesh_confirmation_rate,
            stable_track_rate=stable_track_rate,
            head_face_consistency_pct=head_face_consistency_pct,
            total_detections=total_detections,
            mesh_confirmations=mesh_confirmations,
            stable_tracks=stable_tracks,
            consistent_head_face=consistent_head_face,
            alert_active=alert_active,
            fatigue_alert_active=bool(fatigue_alert_active),
            attention_alert_active=bool(attention_alert_active),
            student_alerts=student_alerts,
            new_alert_event=bool(stored_new_alert_event),
            new_alert_kind=str(new_alert_kind),
            sound_alert_tick=bool(sound_alert_tick),
            decision_explanation=decision_explanation,
            fps=float(self._fps_smooth),
            alert_event_count=int(self.session_alert_event_count),
            valid_observations=int(self.session_valid_observations),
        )

        self._last_track_runtime = next_runtime_snapshot
        self.frame_idx += 1

    def process_external_frame(self, frame_bgr: np.ndarray, want_boxes: bool = False):
        if frame_bgr is None:
            return self.get_stats()

        now = time.time()
        self._update_fps(now)
        self._process_frame_inplace(frame_bgr, now, want_boxes=want_boxes)
        return self.get_stats()

    def get_stats(self) -> dict:
        s = self._last_stats
        return {
            "faces": s.faces,
            "heads": s.heads,
            "class_avg_fatigue_pct": s.class_avg_fatigue_pct,
            "class_avg_attention_pct": s.class_avg_attention_pct,
            "live_quality_pct": s.live_quality_pct,
            "mesh_confirmation_rate": s.mesh_confirmation_rate,
            "stable_track_rate": s.stable_track_rate,
            "head_face_consistency_pct": s.head_face_consistency_pct,
            "total_detections": s.total_detections,
            "mesh_confirmations": s.mesh_confirmations,
            "stable_tracks": s.stable_tracks,
            "consistent_head_face": s.consistent_head_face,
            "alert_active": s.alert_active,
            "fatigue_alert_active": s.fatigue_alert_active,
            "attention_alert_active": s.attention_alert_active,
            "student_alerts": s.student_alerts,
            "new_alert_event": s.new_alert_event,
            "new_alert_kind": s.new_alert_kind,
            "decision_explanation": s.decision_explanation,
            "alert_thresholds": {
                "fatigue_pct": int(getattr(config, "ALERT_FATIGUE_PCT", 50)),
                "attention_min_pct": int(getattr(config, "ALERT_ATTENTION_MIN_PCT", 50)),
            },
            "thresholds": {
                "class_fatigue_on": int(getattr(config, "ALERT_CLASS_FATIGUE_ON", 50)),
                "class_fatigue_off": int(getattr(config, "ALERT_CLASS_FATIGUE_OFF", 45)),
                "class_attention_on": int(getattr(config, "ALERT_CLASS_ATTENTION_ON", 50)),
                "class_attention_off": int(getattr(config, "ALERT_CLASS_ATTENTION_OFF", 55)),
                "student_fatigue_on": int(getattr(config, "ALERT_STUDENT_FATIGUE_ON", 60)),
                "student_fatigue_off": int(getattr(config, "ALERT_STUDENT_FATIGUE_OFF", 52)),
                "student_attention_on": int(getattr(config, "ALERT_STUDENT_ATTENTION_ON", 50)),
                "student_attention_off": int(getattr(config, "ALERT_STUDENT_ATTENTION_OFF", 55)),
                "student_fatigue_critical": int(getattr(config, "ALERT_STUDENT_FATIGUE_CRITICAL", 70)),
                "student_attention_critical": int(getattr(config, "ALERT_STUDENT_ATTENTION_CRITICAL", 30)),
                "min_active_students_for_class_alert": int(
                    getattr(config, "MIN_ACTIVE_STUDENTS_FOR_CLASS_ALERT", 1)
                ),
            },
            "fps": s.fps,
            "faces_data": self._last_faces_data,
            "recent_reports": self._recent_reports,
            "session_summary": self.get_session_summary(),
            "session_active": self.session_active,
            "session_calibrating": self._is_session_calibrating(time.time()),
            "calibration_seconds": float(getattr(config, "SESSION_CALIBRATION_SECONDS", 6.0)),
            "unique_identities_live": int(self.identity_manager.unique_count()),
            "alert_event_count": int(s.alert_event_count),
            "valid_observations": int(s.valid_observations),
            "session_id": self.session_id,
        }