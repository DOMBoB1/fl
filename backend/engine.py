import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import mediapipe as mp
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

import config
from attention import gaze_ratio, attentive_from_gaze
from dataset_store import init_dataset_db
from db import init_db, insert_snapshot, get_recent_snapshots
from face_identity import FaceIdentityManager
from fatigue import perclos_from_events, fatigue_percent
from head_detector import HeadDetector
from multi_face_detector import MultiFaceDetector
from tracker_flow import FlowMultiTracker

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
warnings.filterwarnings("ignore", module="google.protobuf.symbol_database")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

dataset_recorder_ref = None


def set_dataset_recorder(recorder):
    global dataset_recorder_ref
    dataset_recorder_ref = recorder


def lm_xy(face_lms, idx, w, h):
    p = face_lms.landmark[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def expand_bbox(box, w_img, h_img, scale=1.03):
    if box is None:
        return None

    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = int(max(0, cx - bw * 0.5))
    ny1 = int(max(0, cy - bh * 0.5))
    nx2 = int(min(w_img - 1, cx + bw * 0.5))
    ny2 = int(min(h_img - 1, cy + bh * 0.5))

    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return None

    return nx1, ny1, nx2, ny2


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


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    return inter / (area_a + area_b - inter + 1e-9)


def nms_boxes(items, iou_thresh=0.30):
    if not items:
        return []

    ordered = sorted(items, key=lambda x: bbox_area(x["bbox_px"]), reverse=True)
    kept = []

    for item in ordered:
        keep = True
        for existing in kept:
            if item.get("kind") == existing.get("kind") and iou(item["bbox_px"], existing["bbox_px"]) >= iou_thresh:
                keep = False
                break
        if keep:
            kept.append(item)

    return kept


def looks_like_face_box(box, w_img, h_img):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < getattr(config, "FACE_BOX_MIN_SIZE", 16) or bh < getattr(config, "FACE_BOX_MIN_SIZE", 16):
        return False

    if bw > getattr(config, "FACE_BOX_MAX_W_RATIO", 0.72) * w_img:
        return False

    if bh > getattr(config, "FACE_BOX_MAX_H_RATIO", 0.92) * h_img:
        return False

    ratio = bw / (bh + 1e-9)
    if ratio < getattr(config, "FACE_BOX_MIN_ASPECT", 0.30):
        return False
    if ratio > getattr(config, "FACE_BOX_MAX_ASPECT", 1.90):
        return False

    return True


def looks_like_head_box(box, w_img, h_img):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < getattr(config, "HEAD_BOX_MIN_SIZE", 24) or bh < getattr(config, "HEAD_BOX_MIN_SIZE", 24):
        return False

    if bw > getattr(config, "HEAD_BOX_MAX_W_RATIO", 0.82) * w_img:
        return False

    if bh > getattr(config, "HEAD_BOX_MAX_H_RATIO", 0.96) * h_img:
        return False

    ratio = bw / (bh + 1e-9)
    if ratio < getattr(config, "HEAD_BOX_MIN_ASPECT", 0.35):
        return False
    if ratio > getattr(config, "HEAD_BOX_MAX_ASPECT", 1.95):
        return False

    return True


@dataclass
class PersonState:
    perclos_events: List[Tuple[float, bool]] = field(default_factory=list)
    eye_closed: bool = False
    eye_closed_start: Optional[float] = None


@dataclass
class EngineStats:
    faces: int = 0
    heads: int = 0
    class_avg_fatigue_pct: int = 0
    class_avg_attention_pct: int = 0
    alert_active: bool = False
    fps: float = 0.0


class MonitoringEngine:
    def __init__(self):
        self._last_faces_data: List[dict] = []
        self._last_stats = EngineStats()

        self.session_active = False
        self.session_started_at: Optional[float] = None
        self.session_stopped_at: Optional[float] = None
        self.session_samples: List[dict] = []
        self.session_seen_track_ids: Set[int] = set()
        self.session_seen_identity_ids: Set[int] = set()
        self.session_total_face_observations: int = 0

        self.detector = MultiFaceDetector(
            min_conf=config.MIN_DET_CONF if hasattr(config, "MIN_DET_CONF") else 0.4
        )
        self.tracker = FlowMultiTracker(
            ttl_s=config.TRACK_TTL_S,
            iou_t=config.IOU_MATCH_T,
            dist_t=config.CENTER_DIST_T,
            min_pts=config.MIN_TRACK_POINTS,
        )

        self.head_detector = HeadDetector()
        self._next_head_only_id = 10000
        self._cached_head_boxes: List[Tuple[int, int, int, int]] = []
        self._last_head_detect_frame = -1

        self.identity_manager = self._build_identity_manager()
        self.identity_frame_stride = int(getattr(config, "FACE_ID_EVERY_N_FRAMES", 6))
        if self.identity_frame_stride < 1:
            self.identity_frame_stride = 1

        self.track_identity_cache: Dict[int, Optional[int]] = {}

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

        self.people: Dict[int, PersonState] = {}
        self.frame_idx = 0
        self.alert_start: Optional[float] = None

        self._last_time = time.time()
        self._fps_smooth = 0.0

        init_db(config.DB_PATH)
        init_dataset_db()

        self._recent_reports = get_recent_snapshots(
            config.DB_PATH,
            getattr(config, "RECENT_REPORTS_IN_UI", 3),
        )
        self._report_buffer: List[dict] = []
        self._last_report_ts = time.time()

    def _build_identity_manager(self):
        return FaceIdentityManager(
            sim_threshold=float(getattr(config, "FACE_ID_SIM_THRESHOLD", 0.75)),
            min_face_size=int(getattr(config, "FACE_ID_MIN_FACE_SIZE", 90)),
            min_blur_score=float(getattr(config, "FACE_ID_MIN_BLUR_SCORE", 100.0)),
            candidate_hits=int(getattr(config, "FACE_ID_CANDIDATE_HITS", 6)),
            candidate_ttl_s=float(getattr(config, "FACE_ID_CANDIDATE_TTL_S", 3.0)),
        )

    def _reset_live_runtime_state(self):
        self._last_faces_data = []
        self._last_stats = EngineStats()

        self.people = {}
        self.frame_idx = 0
        self.alert_start = None

        self.tracker = FlowMultiTracker(
            ttl_s=config.TRACK_TTL_S,
            iou_t=config.IOU_MATCH_T,
            dist_t=config.CENTER_DIST_T,
            min_pts=config.MIN_TRACK_POINTS,
        )

        self.identity_manager = self._build_identity_manager()
        self.track_identity_cache = {}

        self._cached_head_boxes = []
        self._last_head_detect_frame = -1

        self._last_time = time.time()
        self._fps_smooth = 0.0

    def _reset_session_state(self):
        self.session_started_at = None
        self.session_stopped_at = None
        self.session_samples = []
        self.session_seen_track_ids = set()
        self.session_seen_identity_ids = set()
        self.session_total_face_observations = 0

    def start_session(self):
        self.session_active = True
        self._reset_session_state()
        self._reset_live_runtime_state()
        self.session_started_at = time.time()
        self.session_stopped_at = None

    def stop_session(self):
        self.session_active = False
        self.session_stopped_at = time.time()

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

        unique_so_far = len(self.session_seen_identity_ids)
        if unique_so_far <= 0:
            unique_so_far = len(self.session_seen_track_ids)

        self.session_samples.append(
            {
                "ts": now,
                "faces": int(faces),
                "heads": int(heads),
                "fatigue": float(fatigue),
                "attention": float(attention),
                "alert": int(alert_active),
                "unique_faces_so_far": int(unique_so_far),
                "total_face_observations_so_far": int(self.session_total_face_observations),
            }
        )

    def has_session_data(self) -> bool:
        return len(self.session_samples) > 0

    def get_session_summary(self) -> dict:
        if not self.session_samples:
            return {
                "has_data": False,
                "duration_s": 0,
                "samples": 0,
                "max_faces_seen": 0,
                "max_heads_seen": 0,
                "avg_faces": 0.0,
                "avg_heads": 0.0,
                "avg_fatigue": 0.0,
                "avg_attention": 0.0,
                "alert_count": 0,
                "unique_faces_seen": 0,
                "total_face_observations": 0,
            }

        started = self.session_started_at or self.session_samples[0]["ts"]
        stopped = self.session_stopped_at or self.session_samples[-1]["ts"]
        duration_s = max(0.0, stopped - started)

        faces_vals = [x["faces"] for x in self.session_samples]
        heads_vals = [x["heads"] for x in self.session_samples]
        fatigue_vals = [x["fatigue"] for x in self.session_samples]
        attention_vals = [x["attention"] for x in self.session_samples]
        alert_vals = [x["alert"] for x in self.session_samples]

        unique_faces_seen = len(self.session_seen_identity_ids)
        if unique_faces_seen <= 0:
            unique_faces_seen = len(self.session_seen_track_ids)

        return {
            "has_data": True,
            "started_at": started,
            "stopped_at": stopped,
            "duration_s": duration_s,
            "samples": len(self.session_samples),
            "max_faces_seen": int(max(faces_vals) if faces_vals else 0),
            "max_heads_seen": int(max(heads_vals) if heads_vals else 0),
            "avg_faces": float(np.mean(faces_vals) if faces_vals else 0.0),
            "avg_heads": float(np.mean(heads_vals) if heads_vals else 0.0),
            "avg_fatigue": float(np.mean(fatigue_vals) if fatigue_vals else 0.0),
            "avg_attention": float(np.mean(attention_vals) if attention_vals else 0.0),
            "alert_count": int(sum(alert_vals)),
            "unique_faces_seen": int(unique_faces_seen),
            "total_face_observations": int(self.session_total_face_observations),
        }

    def export_session_report_xlsx(self) -> str:
        summary = self.get_session_summary()
        if not summary["has_data"]:
            raise ValueError("No session data available")

        reports_dir = Path(getattr(config, "SESSION_REPORTS_DIR", "reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = reports_dir / f"session_report_{ts}.xlsx"

        wb = Workbook()
        ws = wb.active
        ws.title = "Session Summary"

        title_fill = PatternFill("solid", fgColor="6F558C")
        left_fill = PatternFill("solid", fgColor="DCE6F1")
        separator_fill = PatternFill("solid", fgColor="C5B0D5")

        white_bold = Font(color="FFFFFF", bold=True)
        black_bold = Font(color="000000", bold=True)

        thin_side = Side(style="thin", color="7F7F7F")
        border_all = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 22

        ws["A1"] = "Class Monitor - Session Report"
        ws["A1"].fill = title_fill
        ws["A1"].font = white_bold
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws["A1"].border = border_all
        ws["B1"].border = border_all
        ws.merge_cells("A1:B1")
        ws.row_dimensions[1].height = 26

        started_str = datetime.fromtimestamp(summary["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
        stopped_str = datetime.fromtimestamp(summary["stopped_at"]).strftime("%Y-%m-%d %H:%M:%S")

        ws["A3"] = "Started at"
        ws["B3"] = started_str
        ws["A4"] = "Stopped at"
        ws["B4"] = stopped_str

        for r in [3, 4]:
            ws[f"A{r}"].fill = left_fill
            ws[f"A{r}"].font = black_bold
            ws[f"A{r}"].alignment = Alignment(horizontal="left", vertical="center")
            ws[f"B{r}"].alignment = Alignment(horizontal="left", vertical="center")
            ws[f"A{r}"].border = border_all
            ws[f"B{r}"].border = border_all

        for cell in ["A5", "B5"]:
            ws[cell].fill = separator_fill
            ws[cell].border = border_all

        metric_rows = [
            ("Session duration (s)", round(summary["duration_s"], 1)),
            ("Samples analyzed", summary["samples"]),
            ("Max faces seen", summary["max_faces_seen"]),
            ("Max heads seen", summary["max_heads_seen"]),
            ("Average faces", round(summary["avg_faces"], 2)),
            ("Average heads", round(summary["avg_heads"], 2)),
            ("Unique faces seen", summary["unique_faces_seen"]),
            ("Total face observations", summary["total_face_observations"]),
            ("Average fatigue (%)", round(summary["avg_fatigue"], 2)),
            ("Average attention (%)", round(summary["avg_attention"], 2)),
            ("Alert count", summary["alert_count"]),
        ]

        start_row = 6
        for i, (label, value) in enumerate(metric_rows, start=start_row):
            ws[f"A{i}"] = label
            ws[f"B{i}"] = value
            ws[f"A{i}"].fill = left_fill
            ws[f"A{i}"].font = black_bold
            ws[f"A{i}"].alignment = Alignment(horizontal="left", vertical="center")
            ws[f"B{i}"].alignment = Alignment(horizontal="right", vertical="center")
            ws[f"A{i}"].border = border_all
            ws[f"B{i}"].border = border_all

        ws2 = wb.create_sheet("Session Samples")
        headers = [
            "Timestamp",
            "Faces",
            "Heads",
            "Fatigue (%)",
            "Attention (%)",
            "Alert",
            "Unique faces so far",
            "Total face observations so far",
        ]
        header_fill = PatternFill("solid", fgColor="26B6DE")

        for col, header in enumerate(headers, start=1):
            cell = ws2.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = white_bold
            cell.border = border_all
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for idx, sample in enumerate(self.session_samples, start=2):
            ws2.cell(row=idx, column=1, value=datetime.fromtimestamp(sample["ts"]).strftime("%Y-%m-%d %H:%M:%S"))
            ws2.cell(row=idx, column=2, value=sample["faces"])
            ws2.cell(row=idx, column=3, value=sample["heads"])
            ws2.cell(row=idx, column=4, value=round(sample["fatigue"], 2))
            ws2.cell(row=idx, column=5, value=round(sample["attention"], 2))
            ws2.cell(row=idx, column=6, value=sample["alert"])
            ws2.cell(row=idx, column=7, value=sample["unique_faces_so_far"])
            ws2.cell(row=idx, column=8, value=sample["total_face_observations_so_far"])

            for c in range(1, 9):
                ws2.cell(row=idx, column=c).border = border_all

        ws2.column_dimensions["A"].width = 22
        ws2.column_dimensions["B"].width = 12
        ws2.column_dimensions["C"].width = 12
        ws2.column_dimensions["D"].width = 14
        ws2.column_dimensions["E"].width = 16
        ws2.column_dimensions["F"].width = 10
        ws2.column_dimensions["G"].width = 18
        ws2.column_dimensions["H"].width = 28

        wb.save(out_path)
        return str(out_path)

    def _update_fps(self, now: float):
        dt = now - self._last_time
        self._last_time = now

        fps_inst = (1.0 / dt) if dt > 1e-6 else 0.0
        self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps_inst

    def _maybe_store_report(self, now: float, faces: int, heads: int, fatigue: int, attention: int, alert_active: bool):
        self._report_buffer.append(
            {
                "ts": now,
                "faces": float(faces),
                "heads": float(heads),
                "fatigue": float(fatigue),
                "attention": float(attention),
                "alert": int(alert_active),
            }
        )

        interval_s = int(getattr(config, "REPORT_INTERVAL_S", 60))
        if interval_s <= 0:
            interval_s = 60

        if now - self._last_report_ts < interval_s:
            return

        if not self._report_buffer:
            self._last_report_ts = now
            return

        faces_avg = float(np.mean([x["faces"] for x in self._report_buffer]))
        fatigue_avg = float(np.mean([x["fatigue"] for x in self._report_buffer]))
        attention_avg = float(np.mean([x["attention"] for x in self._report_buffer]))
        alert_count = int(sum(x["alert"] for x in self._report_buffer))
        max_fatigue = float(np.max([x["fatigue"] for x in self._report_buffer]))
        min_attention = float(np.min([x["attention"] for x in self._report_buffer]))

        insert_snapshot(
            config.DB_PATH,
            created_at=now,
            interval_s=interval_s,
            faces_avg=faces_avg,
            fatigue_avg=fatigue_avg,
            attention_avg=attention_avg,
            alert_count=alert_count,
            max_fatigue=max_fatigue,
            min_attention=min_attention,
        )

        self._recent_reports = get_recent_snapshots(
            config.DB_PATH,
            getattr(config, "RECENT_REPORTS_IN_UI", 3),
        )

        self._report_buffer = []
        self._last_report_ts = now

    def _save_dataset_sample(
        self,
        frame_bgr: np.ndarray,
        visible_face_track_ids_this_frame: List[int],
        head_count_this_frame: int,
        all_faces_for_boxes: List[dict],
    ):
        if dataset_recorder_ref is None:
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
                }
                for item in all_faces_for_boxes
            ],
        }

        dataset_recorder_ref.maybe_save(
            frame=frame_bgr.copy(),
            extra=extra_payload,
        )

    def _best_face_inside_head(self, frame_bgr: np.ndarray, head_box: Tuple[int, int, int, int]):
        h_img, w_img = frame_bgr.shape[:2]
        expanded = expand_bbox(head_box, w_img, h_img, scale=1.08)
        if expanded is None:
            expanded = head_box

        x1, y1, x2, y2 = expanded
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        dets = self.detector.detect(roi, detect_width=config.DETECT_WIDTH)
        if not dets:
            return None

        best = None
        best_score = -1.0

        for fx1, fy1, fx2, fy2 in dets:
            gbox = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
            if not looks_like_face_box(gbox, w_img, h_img):
                continue

            overlap = iou(gbox, head_box)
            if overlap < getattr(config, "FACE_INSIDE_HEAD_MIN_IOU", 0.03):
                continue

            score = bbox_area(gbox) + overlap * 10000.0
            if score > best_score:
                best_score = score
                best = gbox

        return best

    def _process_frame_inplace(self, frame_bgr: np.ndarray, now: float, want_boxes: bool) -> None:
        self.tracker.update_optical_flow(frame_bgr, now)

        detect_every = int(getattr(config, "DETECT_EVERY_N_FRAMES", 3))
        if detect_every < 1:
            detect_every = 1

        if self.frame_idx % detect_every == 0:
            dets = self.detector.detect(frame_bgr, detect_width=config.DETECT_WIDTH)
            self.tracker.sync_with_detections(frame_bgr, dets, now)

        self.tracker.drop_stale(now)
        tracks = self.tracker.get_tracks(frame_bgr)

        for tid in tracks.keys():
            if tid not in self.people:
                self.people[tid] = PersonState()

        active_fatigue: List[int] = []
        active_attention: List[float] = []

        confirmed_faces: List[dict] = []
        confirmed_track_ids_this_frame: List[int] = []
        visible_face_track_ids_this_frame: List[int] = []
        head_boxes_for_display: List[dict] = []

        h_img, w_img = frame_bgr.shape[:2]
        face_mesh_scale = float(getattr(config, "FACE_MESH_EXPAND_SCALE", 1.18))

        if self.head_detector.enabled:
            head_stride = int(getattr(config, "HEAD_DETECT_EVERY_N_FRAMES", 5))
            if head_stride < 1:
                head_stride = 1

            if (
                self._last_head_detect_frame < 0
                or (self.frame_idx - self._last_head_detect_frame) >= head_stride
            ):
                self._cached_head_boxes = self.head_detector.detect(frame_bgr)
                self._last_head_detect_frame = self.frame_idx

        cached_heads = []
        for hb in self._cached_head_boxes:
            if looks_like_head_box(hb, w_img, h_img):
                self._next_head_only_id += 1
                cached_heads.append(
                    {
                        "id": self._next_head_only_id,
                        "bbox_px": hb,
                        "kind": "head",
                    }
                )

        head_boxes_for_display = cached_heads[:]

        # Regular face tracks from tracker
        for tid, (x1, y1, x2, y2) in tracks.items():
            x1 = max(0, min(w_img - 1, int(x1)))
            y1 = max(0, min(h_img - 1, int(y1)))
            x2 = max(0, min(w_img - 1, int(x2)))
            y2 = max(0, min(h_img - 1, int(y2)))

            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            raw_box = (x1, y1, x2, y2)
            if not looks_like_face_box(raw_box, w_img, h_img):
                continue

            visible_face_track_ids_this_frame.append(int(tid))

            identity_crop_box = expand_bbox(raw_box, w_img, h_img, scale=1.06)
            if identity_crop_box is None:
                identity_crop_box = raw_box

            ix1, iy1, ix2, iy2 = identity_crop_box
            identity_face = frame_bgr[iy1:iy2, ix1:ix2]
            if identity_face.size == 0:
                continue

            face_h, face_w = identity_face.shape[:2]
            if face_w < getattr(config, "FACE_BOX_MIN_SIZE", 16) or face_h < getattr(config, "FACE_BOX_MIN_SIZE", 16):
                continue

            identity_id = self.track_identity_cache.get(int(tid))
            if self.frame_idx % self.identity_frame_stride == 0:
                new_identity_id = self.identity_manager.update_track(int(tid), identity_face, now)
                if new_identity_id is not None:
                    identity_id = int(new_identity_id)
                    self.track_identity_cache[int(tid)] = identity_id

            if identity_id is not None and self.session_active:
                self.session_seen_identity_ids.add(int(identity_id))

            mesh_box = expand_bbox(raw_box, w_img, h_img, scale=face_mesh_scale)
            if mesh_box is not None:
                mx1, my1, mx2, my2 = mesh_box
                mesh_face = frame_bgr[my1:my2, mx1:mx2]
                if mesh_face.size != 0:
                    mesh_h, mesh_w = mesh_face.shape[:2]
                    rgb = np.ascontiguousarray(mesh_face[:, :, ::-1])
                    res = self.mesh.process(rgb)
                    has_landmarks = bool(res and res.multi_face_landmarks)

                    if has_landmarks:
                        lms = res.multi_face_landmarks[0]

                        ear_l = eye_aspect_ratio(lms, LEFT_EYE, mesh_w, mesh_h)
                        ear_r = eye_aspect_ratio(lms, RIGHT_EYE, mesh_w, mesh_h)
                        ear = (ear_l + ear_r) / 2.0
                        is_closed = ear < config.EYES_CLOSED_EAR_T

                        st = self.people[tid]
                        st.perclos_events.append((now, is_closed))
                        st.perclos_events = [
                            (t, c)
                            for (t, c) in st.perclos_events
                            if t >= now - (config.PERCLOS_WINDOW_S + 2)
                        ]

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
                        active_fatigue.append(int(fat_pct))

                        gz = gaze_ratio(lms, mesh_w, mesh_h)
                        att_score = attentive_from_gaze(gz, config.LOOK_MIN, config.LOOK_MAX)
                        active_attention.append(float(att_score))

                        confirmed_track_ids_this_frame.append(int(tid))

            display_id = int(identity_id) if identity_id is not None else int(tid)
            if want_boxes:
                fb = expand_bbox(raw_box, w_img, h_img, scale=1.06)
                if fb is not None and looks_like_face_box(fb, w_img, h_img):
                    confirmed_faces.append(
                        {
                            "id": display_id,
                            "bbox_px": fb,
                            "kind": "face",
                        }
                    )

        # Head-first local face search
        for head_item in cached_heads:
            head_box = head_item["bbox_px"]
            found_face = self._best_face_inside_head(frame_bgr, head_box)
            if found_face is None:
                continue

            if want_boxes:
                confirmed_faces.append(
                    {
                        "id": head_item["id"],
                        "bbox_px": found_face,
                        "kind": "face",
                    }
                )

        self.identity_manager.cleanup(now, confirmed_track_ids_this_frame)

        active_ids = set(int(x) for x in confirmed_track_ids_this_frame)
        for tid in list(self.track_identity_cache.keys()):
            if tid not in active_ids:
                del self.track_identity_cache[tid]

        if self.session_active and visible_face_track_ids_this_frame:
            self.session_seen_track_ids.update(visible_face_track_ids_this_frame)
            self.session_total_face_observations += len(visible_face_track_ids_this_frame)

        all_boxes_for_display = head_boxes_for_display + confirmed_faces
        all_boxes_for_display = nms_boxes(all_boxes_for_display, iou_thresh=0.30)

        if want_boxes:
            self._last_faces_data = [
                {
                    "id": item["id"],
                    "kind": item.get("kind", "face"),
                    "bbox_n": [
                        item["bbox_px"][0] / w_img,
                        item["bbox_px"][1] / h_img,
                        item["bbox_px"][2] / w_img,
                        item["bbox_px"][3] / h_img,
                    ],
                }
                for item in all_boxes_for_display
            ]
        else:
            self._last_faces_data = []

        face_count = len([x for x in all_boxes_for_display if x.get("kind") == "face"])
        head_count = len([x for x in all_boxes_for_display if x.get("kind") == "head"])

        faces_for_stats = face_count
        if faces_for_stats == 0 and getattr(config, "COUNT_HEADS_WHEN_NO_FACE", True):
            faces_for_stats = head_count

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

        self._append_session_sample(
            now=now,
            faces=faces_for_stats,
            heads=head_count,
            fatigue=class_avg_fatigue,
            attention=class_avg_attention,
            alert_active=alert_active,
        )

        self._maybe_store_report(
            now=now,
            faces=faces_for_stats,
            heads=head_count,
            fatigue=class_avg_fatigue,
            attention=class_avg_attention,
            alert_active=alert_active,
        )

        self._save_dataset_sample(
            frame_bgr=frame_bgr,
            visible_face_track_ids_this_frame=visible_face_track_ids_this_frame,
            head_count_this_frame=head_count,
            all_faces_for_boxes=all_boxes_for_display,
        )

        self._last_stats = EngineStats(
            faces=faces_for_stats,
            heads=head_count,
            class_avg_fatigue_pct=class_avg_fatigue,
            class_avg_attention_pct=class_avg_attention,
            alert_active=alert_active,
            fps=float(self._fps_smooth),
        )

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
            "alert_active": s.alert_active,
            "fps": s.fps,
            "faces_data": self._last_faces_data,
            "recent_reports": self._recent_reports,
            "session_summary": self.get_session_summary(),
            "session_active": self.session_active,
            "unique_identities_live": int(self.identity_manager.unique_count()),
        }