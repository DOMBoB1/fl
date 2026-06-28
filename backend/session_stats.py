from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

import numpy as np


class SessionStatsManager:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.session_started_at: Optional[float] = None
        self.session_stopped_at: Optional[float] = None
        self.session_id: Optional[int] = None
        self.session_samples: List[Dict[str, Any]] = []
        self.session_seen_track_ids: Set[int] = set()
        self.session_seen_identity_ids: Set[int] = set()
        self.session_total_face_observations: int = 0
        self.session_valid_observations: int = 0
        self.session_alert_event_count: int = 0
        self.session_student_stats: Dict[str, Dict[str, Any]] = {}

    def start(self, session_id: Optional[int] = None, started_at: Optional[float] = None) -> None:
        self.reset()
        self.session_id = session_id
        self.session_started_at = float(started_at if started_at is not None else time.time())
        self.session_stopped_at = None

    def stop(self, stopped_at: Optional[float] = None) -> None:
        self.session_stopped_at = float(stopped_at if stopped_at is not None else time.time())

    def update_student_stats(
        self,
        student_key: str,
        fatigue_pct: float,
        attention_pct: float,
        alert_type: str = "none",
        severity: str = "none",
        reason: str = "",
        decision: str = "",
    ) -> None:
        item = self.session_student_stats.get(student_key)
        if item is None:
            item = {
                "student_id": str(student_key),
                "fatigue_sum": 0.0,
                "attention_sum": 0.0,
                "samples": 0,
                "fatigue_max": 0.0,
                "attention_min": 100.0,
                "final_alert_type": "none",
                "final_severity": "none",
                "reason": "",
                "decision": "",
            }
            self.session_student_stats[student_key] = item

        item["fatigue_sum"] += float(fatigue_pct)
        item["attention_sum"] += float(attention_pct)
        item["samples"] += 1
        item["fatigue_max"] = max(float(item["fatigue_max"]), float(fatigue_pct))
        item["attention_min"] = min(float(item["attention_min"]), float(attention_pct))

        sev_rank = {"none": 0, "warning": 1, "critical": 2}
        if sev_rank.get(str(severity), 0) >= sev_rank.get(str(item["final_severity"]), 0):
            item["final_severity"] = str(severity or "none")
            item["final_alert_type"] = str(alert_type or "none")
            item["reason"] = str(reason or "")
            item["decision"] = str(decision or "")

    def build_students_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        MIN_SAMPLES = 15

        for student_key, item in self.session_student_stats.items():
            samples = int(item.get("samples", 0))

            if samples < MIN_SAMPLES:
                continue

            fatigue_avg = float(item.get("fatigue_sum", 0.0)) / samples
            attention_avg = float(item.get("attention_sum", 0.0)) / samples
            fatigue_max = float(item.get("fatigue_max", 0.0))
            attention_min = float(item.get("attention_min", 100.0))

            final_alert_type = str(item.get("final_alert_type", "none") or "none")
            final_severity = str(item.get("final_severity", "none") or "none")
            reason = str(item.get("reason", "") or "")
            decision = str(item.get("decision", "") or "")

            rows.append(
                {
                    "student_id": str(student_key),
                    "fatigue_avg": round(fatigue_avg, 2),
                    "attention_avg": round(attention_avg, 2),
                    "fatigue_max": round(fatigue_max, 2),
                    "attention_min": round(attention_min, 2),
                    "final_alert_type": final_alert_type,
                    "final_severity": final_severity,
                    "reason": reason,
                    "decision": decision,
                    "samples": samples,
                }
            )

        rows.sort(
            key=lambda x: (
                0 if x["final_severity"] == "critical" else 1 if x["final_severity"] == "warning" else 2,
                x["attention_min"],
                -x["fatigue_max"],
            )
        )

        return rows

    def append_sample(
        self,
        now: float,
        faces: int,
        heads: int,
        fatigue: float,
        attention: float,
        alert_active: bool,
    ) -> None:
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
                "valid_observations_so_far": int(self.session_valid_observations),
                "alert_events_so_far": int(self.session_alert_event_count),
            }
        )

    def has_data(self) -> bool:
        return len(self.session_samples) > 0

    def get_summary(self) -> Dict[str, Any]:
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
                "alert_event_count": 0,
                "unique_faces_seen": 0,
                "total_face_observations": 0,
                "valid_observations": 0,
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
            "alert_event_count": int(self.session_alert_event_count),
            "unique_faces_seen": int(unique_faces_seen),
            "total_face_observations": int(self.session_total_face_observations),
            "valid_observations": int(self.session_valid_observations),
        }
