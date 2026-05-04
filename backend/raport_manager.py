from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import config
from db import get_recent_snapshots, insert_snapshot, update_session_report_path as update_session_raport_path


def _border_subtire() -> Border:
    side = Side(style="thin", color="A6A6A6")
    return Border(left=side, right=side, top=side, bottom=side)


def _stil_titlu(cell, fill: str = "6F5B95") -> None:
    cell.font = Font(color="FFFFFF", bold=True, size=12)
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border = _border_subtire()


def _stil_antet(cell, fill: str = "20B7D7") -> None:
    cell.font = Font(color="FFFFFF", bold=True)
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border = _border_subtire()


def _stil_nota(cell, fill: str = "FFF2CC") -> None:
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.font = Font(color="000000", italic=True)
    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    cell.border = _border_subtire()


def _latime_automata(ws, min_width: int = 12, max_width: int = 40) -> None:
    for col_cells in ws.columns:
        first = col_cells[0]
        if getattr(first, "column", None) is None:
            continue
        col_letter = get_column_letter(first.column)
        max_len = 0
        for cell in col_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 2, max_width))


def _fill_stare(stare: str) -> PatternFill:
    stare = (stare or "").lower()
    if stare == "critical":
        return PatternFill("solid", fgColor="F4CCCC")
    if stare == "warning":
        return PatternFill("solid", fgColor="FCE5CD")
    if stare in {"fatigue risk", "attention drop"}:
        return PatternFill("solid", fgColor="FFF2CC")
    return PatternFill("solid", fgColor="D9EAD3")


def _rotunjeste(value: Any, digits: int = 2) -> Any:
    try:
        return round(float(value), digits)
    except Exception:
        return value


def _student_id_excel_value(value: Any) -> Any:
    text = str(value).strip()
    if text.isdigit():
        try:
            return int(text)
        except Exception:
            return text
    return text


def _incarca_metrici_evaluare() -> Dict[str, Any]:
    metrics_path = Path(
        getattr(
            config,
            "EVAL_METRICS_PATH",
            Path(getattr(config, "SESSION_REPORTS_DIR", "reports")) / "last_eval_metrics.json",
        )
    )

    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class RaportManager:
    def __init__(
        self,
        db_path: str,
        recent_rapoarte_limit: int = 3,
        interval_raport_s: int = 60,
    ) -> None:
        self.db_path = db_path
        self.recent_rapoarte_limit = int(recent_rapoarte_limit)
        self.interval_raport_s = int(interval_raport_s)
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.raport_buffer: List[Dict[str, Any]] = []
        self.trend_buffer: List[Dict[str, Any]] = []
        self.last_raport_ts: float = time.time()
        self.recent_rapoarte: List[Dict[str, Any]] = []

    def bind_session(self, session_id: Optional[int]) -> None:
        self.reset_runtime()
        if session_id is None:
            return
        self.recent_rapoarte = get_recent_snapshots(
            self.db_path,
            session_id=session_id,
            limit=self.recent_rapoarte_limit,
        )

    def adauga_esantion_runtime(
        self,
        now: float,
        faces: int,
        heads: int,
        fatigue: float,
        attention: float,
        fatigue_alert_active: bool,
        attention_alert_active: bool,
        student_alerts: List[dict],
        studenti_runtime: List[dict],
        class_decision_explanation: str,
    ) -> None:
        self.raport_buffer.append(
            {
                "ts": float(now),
                "faces": float(faces),
                "heads": float(heads),
                "fatigue": float(fatigue),
                "attention": float(attention),
                "fatigue_alert_active": bool(fatigue_alert_active),
                "attention_alert_active": bool(attention_alert_active),
                "student_alert_count": int(len(student_alerts or [])),
                "critical_student_count": int(
                    sum(1 for x in (student_alerts or []) if str(x.get("severity", "")) == "critical")
                ),
                "decision_explanation": str(class_decision_explanation or ""),
            }
        )

        puncte_studenti: List[Dict[str, Any]] = []
        vazuti: Set[str] = set()
        for item in studenti_runtime or []:
            student_id = str(item.get("student_id", "")).strip()
            if not student_id or student_id in vazuti:
                continue
            vazuti.add(student_id)
            puncte_studenti.append(
                {
                    "student_id": student_id,
                    "fatigue_pct": _rotunjeste(item.get("fatigue_pct", 0.0)),
                    "attention_pct": _rotunjeste(item.get("attention_pct", 0.0)),
                }
            )

        self.trend_buffer.append(
            {
                "ts": float(now),
                "timestamp_label": datetime.fromtimestamp(float(now)).strftime("%H:%M:%S"),
                "studenti": puncte_studenti,
            }
        )

    def poate_stoca_snapshot(
        self,
        session_id: Optional[int],
        now: float,
        class_alert_type_and_level_fn: Callable[[bool, bool, int], Tuple[str, str]],
        class_reason_and_decision_fn: Callable[[float, float, bool, bool, int, str], Tuple[str, str]],
    ) -> None:
        if session_id is None:
            return

        interval_s = self.interval_raport_s if self.interval_raport_s > 0 else 60
        if now - self.last_raport_ts < interval_s:
            return

        if not self.raport_buffer:
            self.last_raport_ts = now
            return

        faces_avg = float(np.mean([x["faces"] for x in self.raport_buffer]))
        heads_avg = float(np.mean([x["heads"] for x in self.raport_buffer]))
        fatigue_avg = float(np.mean([x["fatigue"] for x in self.raport_buffer]))
        attention_avg = float(np.mean([x["attention"] for x in self.raport_buffer]))
        fatigue_max = float(np.max([x["fatigue"] for x in self.raport_buffer]))
        attention_min = float(np.min([x["attention"] for x in self.raport_buffer]))
        student_alert_count = int(max(x["student_alert_count"] for x in self.raport_buffer))
        critical_student_count = int(max(x["critical_student_count"] for x in self.raport_buffer))

        last = self.raport_buffer[-1]
        class_alert_type, class_alert_level = class_alert_type_and_level_fn(
            bool(last["fatigue_alert_active"]),
            bool(last["attention_alert_active"]),
            int(round(faces_avg)),
        )
        reason, decision = class_reason_and_decision_fn(
            fatigue_avg,
            attention_avg,
            bool(last["fatigue_alert_active"]),
            bool(last["attention_alert_active"]),
            int(round(faces_avg)),
            str(last.get("decision_explanation", "")),
        )

        insert_snapshot(
            db_path=self.db_path,
            session_id=session_id,
            created_at=now,
            faces_avg=faces_avg,
            heads_avg=heads_avg,
            active_students_avg=faces_avg,
            fatigue_avg=fatigue_avg,
            attention_avg=attention_avg,
            fatigue_max=fatigue_max,
            attention_min=attention_min,
            class_alert_type=class_alert_type,
            class_alert_level=class_alert_level,
            student_alert_count=student_alert_count,
            critical_student_count=critical_student_count,
            reason=reason,
            decision=decision,
        )

        self.recent_rapoarte = get_recent_snapshots(
            self.db_path,
            session_id=session_id,
            limit=self.recent_rapoarte_limit,
        )

        self.raport_buffer = []
        self.last_raport_ts = now

    def _stare_din_student(self, student: Dict[str, Any]) -> str:
        severity = str(student.get("final_severity", "none") or "none").lower()
        alert_type = str(student.get("final_alert_type", "none") or "none").lower()
        attention = float(student.get("attention_avg", 100.0) or 100.0)
        fatigue = float(student.get("fatigue_avg", 0.0) or 0.0)

        if attention < 30 and fatigue > 60:
            return "Critical"
        if severity == "critical" or severity == "warning":
            return "Warning"
        if alert_type == "fatigue":
            return "Fatigue risk"
        if alert_type == "attention":
            return "Attention drop"
        return "Monitoring"

    def _actiune_recomandata(self, student: Dict[str, Any]) -> str:
        severity = str(student.get("final_severity", "none") or "none").lower()
        alert_type = str(student.get("final_alert_type", "none") or "none").lower()

        if severity == "critical":
            return "Individual follow-up is recommended"
        if severity == "warning":
            return "Closer monitoring is recommended"
        if alert_type == "fatigue":
            return "Monitor for fatigue-related signs"
        if alert_type == "attention":
            return "Apply attention refocusing strategies"
        return "Continue regular monitoring"

    def _humanize_reason(self, reason: str, attention: float, fatigue: float) -> str:
        text = str(reason or "").strip().lower()

        if attention <= 20:
            return "Student had major difficulty staying focused throughout the session."
        if attention <= 40:
            return "Student seemed distracted for a significant part of the session."
        if attention <= 60:
            return "Student showed fluctuating attention during the session."

        if fatigue >= 80:
            return "Student showed strong signs of fatigue and low energy."
        if fatigue >= 60:
            return "Student appeared tired during several moments of the session."
        if fatigue >= 40:
            return "Student showed mild signs of fatigue."

        if "critical" in text:
            return "Student struggled significantly to remain engaged."
        if "inattentive" in text:
            return "Student seemed distracted quite often during the session."
        if "fatigue" in text or "tired" in text or "sleepy" in text:
            return "Student showed visible signs of tiredness."
        if "attention" in text or "focus" in text:
            return "Student had some difficulty maintaining consistent attention."

        if text:
            cleaned = reason.strip()
            return cleaned[:1].upper() + cleaned[1:] if cleaned else "No major issues observed."

        return "No major issues observed."

    def _student_ids_valizi_pentru_raport(
        self,
        session_students: List[Dict[str, Any]],
    ) -> List[str]:
        min_samples = int(getattr(config, "MIN_SAMPLES_FOR_REPORT", 15))
        min_visible_ratio = float(getattr(config, "MIN_VISIBLE_RATIO_FOR_REPORT", 0.15))
        min_span_seconds = float(getattr(config, "MIN_SPAN_SECONDS_FOR_REPORT", 8.0))

        total_rows = len(self.trend_buffer)
        if total_rows <= 0:
            ids = []
            for student in session_students or []:
                sid = str(student.get("student_id", "")).strip()
                if sid:
                    ids.append(sid)
            return sorted(set(ids), key=lambda x: int(x) if x.isdigit() else x)

        per_student: Dict[str, Dict[str, Any]] = {}

        for punct in self.trend_buffer:
            ts = float(punct.get("ts", 0.0))
            for item in punct.get("studenti", []) or []:
                sid = str(item.get("student_id", "")).strip()
                if not sid:
                    continue

                entry = per_student.setdefault(
                    sid,
                    {
                        "samples": 0,
                        "first_ts": None,
                        "last_ts": None,
                    },
                )

                has_metric = (
                    item.get("fatigue_pct") is not None
                    or item.get("attention_pct") is not None
                )
                if not has_metric:
                    continue

                entry["samples"] += 1
                if entry["first_ts"] is None:
                    entry["first_ts"] = ts
                entry["last_ts"] = ts

        valid_ids: List[str] = []

        for sid, info in per_student.items():
            samples = int(info.get("samples", 0))
            visible_ratio = samples / total_rows if total_rows > 0 else 0.0

            first_ts = info.get("first_ts")
            last_ts = info.get("last_ts")
            span_seconds = 0.0
            if first_ts is not None and last_ts is not None:
                span_seconds = max(0.0, float(last_ts) - float(first_ts))

            if (
                samples >= min_samples
                and visible_ratio >= min_visible_ratio
                and span_seconds >= min_span_seconds
            ):
                valid_ids.append(sid)

        if not valid_ids and per_student:
            best_sid = max(
                per_student.items(),
                key=lambda kv: (
                    int(kv[1].get("samples", 0)),
                    float((kv[1].get("last_ts") or 0.0)) - float((kv[1].get("first_ts") or 0.0)),
                ),
            )[0]
            valid_ids = [best_sid]

        return sorted(set(valid_ids), key=lambda x: int(x) if x.isdigit() else x)

    def _filtreaza_studenti_sumar(
        self,
        session_students: List[Dict[str, Any]],
        valid_student_ids: List[str],
    ) -> List[Dict[str, Any]]:
        valid_set = set(valid_student_ids)
        if not valid_set:
            return []

        return [
            student
            for student in (session_students or [])
            if str(student.get("student_id", "")).strip() in valid_set
        ]

    def _construieste_sheet_overview(self, wb: Workbook, session_summary: Dict[str, Any]) -> None:
        ws = wb.active
        ws.title = "Session Overview"
        ws.sheet_view.showGridLines = True
        ws.freeze_panes = "A3"

        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 78

        ws.merge_cells("A1:B1")
        ws["A1"] = "Class Monitor - Session Overview"
        _stil_titlu(ws["A1"])

        eval_metrics = _incarca_metrici_evaluare()

        rows = [
            ("Started at", session_summary.get("started_at", "")),
            ("Stopped at", session_summary.get("stopped_at", "")),
            ("Duration (minutes)", _rotunjeste(float(session_summary.get("duration_s", 0.0)) / 60.0, 2)),
            ("Unique faces seen", int(session_summary.get("unique_faces_seen", 0))),
            ("Max faces seen", int(session_summary.get("max_faces_seen", 0))),
            ("Max heads seen", int(session_summary.get("max_heads_seen", 0))),
            ("Avg faces", _rotunjeste(session_summary.get("avg_faces", 0.0))),
            ("Avg heads", _rotunjeste(session_summary.get("avg_heads", 0.0))),
            ("Avg fatigue (%)", _rotunjeste(session_summary.get("avg_fatigue", 0.0))),
            (
                "Avg attention (%)",
                _rotunjeste(
                    float(np.mean([x.get("attention", 0.0) for x in self.raport_buffer]))
                    if self.raport_buffer
                    else session_summary.get("avg_attention", 0.0)
                ),
            ),
            ("Detection accuracy (%)", _rotunjeste(eval_metrics.get("accuracy_pct", ""))),
            ("Precision (%)", _rotunjeste(eval_metrics.get("precision_pct", ""))),
            ("Recall (%)", _rotunjeste(eval_metrics.get("recall_pct", ""))),
            ("F1 score (%)", _rotunjeste(eval_metrics.get("f1_pct", ""))),
            ("Moments needing attention", int(session_summary.get("alert_event_count", session_summary.get("alert_count", 0)))),
        ]

        start_row = 3
        for idx, (label, value) in enumerate(rows, start=start_row):
            ws.cell(idx, 1, label)
            ws.cell(idx, 2, value)

            ws.cell(idx, 1).font = Font(bold=True)
            ws.cell(idx, 1).fill = PatternFill("solid", fgColor="D9E2F3")
            ws.cell(idx, 1).border = _border_subtire()
            ws.cell(idx, 2).border = _border_subtire()
            ws.cell(idx, 2).alignment = Alignment(horizontal="right", vertical="center")

        note_row = start_row + len(rows)
        ws.cell(note_row, 1, "What this means")
        ws.cell(
            note_row,
            2,
            "This counts the moments when the monitored student or group showed clearer signs of fatigue or low attention. A higher value means there were more situations that may have needed intervention, a short break, or a change in teaching pace.",
        )

        ws.cell(note_row, 1).font = Font(bold=True)
        ws.cell(note_row, 1).fill = PatternFill("solid", fgColor="FFF2CC")
        ws.cell(note_row, 2).fill = PatternFill("solid", fgColor="FFF2CC")
        ws.cell(note_row, 1).border = _border_subtire()
        ws.cell(note_row, 2).border = _border_subtire()
        ws.cell(note_row, 1).alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        ws.cell(note_row, 2).alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        ws.row_dimensions[note_row].height = 72

    def _construieste_sheet_studenti(
        self,
        wb: Workbook,
        session_students: List[Dict[str, Any]],
        valid_student_ids: List[str],
    ) -> None:
        ws = wb.create_sheet("Students Requiring Attention")
        ws.sheet_view.showGridLines = True
        ws.freeze_panes = "A3"

        ws.merge_cells("A1:H1")
        ws["A1"] = "Students Requiring Attention"
        _stil_titlu(ws["A1"])

        headers = [
            "Student ID",
            "Fatigue Avg",
            "Attention Avg",
            "Fatigue Max",
            "Attention Min",
            "Status",
            "Recommended Action",
            "Observation",
        ]

        for col_idx, header in enumerate(headers, start=1):
            _stil_antet(ws.cell(3, col_idx, header))

        rand = 4
        studenti_vizibili = []
        valid_set = set(valid_student_ids)
        display_student_ids = {str(sid): idx for idx, sid in enumerate(valid_student_ids, start=1)}

        for student in session_students or []:
            sid = str(student.get("student_id", "")).strip()
            if sid not in valid_set:
                continue

            severity = str(student.get("final_severity", "none") or "none").lower()
            alert_type = str(student.get("final_alert_type", "none") or "none").lower()
            if severity == "none" and alert_type == "none":
                continue

            studenti_vizibili.append(student)

        if not studenti_vizibili:
            ws.merge_cells("A4:H4")
            ws["A4"] = "No students required attention in this session."
            ws["A4"].alignment = Alignment(horizontal="center")
            ws["A4"].border = _border_subtire()
            _latime_automata(ws, min_width=14, max_width=40)
            return

        for student in studenti_vizibili:
            stare = self._stare_din_student(student)
            actiune = self._actiune_recomandata(student)
            observatie = self._humanize_reason(
                str(student.get("reason", "") or ""),
                float(student.get("attention_avg", 0.0) or 0.0),
                float(student.get("fatigue_avg", 0.0) or 0.0),
            )

            valori = [
                display_student_ids.get(str(student.get("student_id", "")).strip(), rand - 3),
                _rotunjeste(student.get("fatigue_avg", 0.0)),
                _rotunjeste(student.get("attention_avg", 0.0)),
                _rotunjeste(student.get("fatigue_max", 0.0)),
                _rotunjeste(student.get("attention_min", 0.0)),
                stare,
                actiune,
                observatie,
            ]

            for col_idx, value in enumerate(valori, start=1):
                cell = ws.cell(rand, col_idx, value)
                cell.border = _border_subtire()

                if col_idx == 1:
                    if isinstance(value, int):
                        cell.number_format = "0"
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif col_idx in {2, 3, 4, 5}:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.number_format = "0.00"
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

            fill = _fill_stare(stare)
            for col_idx in range(1, len(headers) + 1):
                ws.cell(rand, col_idx).fill = fill

            rand += 1

        widths = {
            "A": 14,
            "B": 14,
            "C": 16,
            "D": 14,
            "E": 14,
            "F": 14,
            "G": 32,
            "H": 42,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

    def _construieste_sheet_date_trend(
        self,
        wb: Workbook,
        valid_student_ids: List[str],
    ) -> Tuple[List[str], List[str], int]:
        ws = wb.create_sheet("Ignore - Trend Data")
        ws.sheet_view.showGridLines = True

        student_ids = sorted(
            [str(x).strip() for x in valid_student_ids if str(x).strip()],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        timestamps = [punct["timestamp_label"] for punct in self.trend_buffer]
        student_id_set = set(student_ids)
        display_student_ids = {str(sid): idx for idx, sid in enumerate(student_ids, start=1)}

        ws.merge_cells("A1:N1")
        ws["A1"] = "Ignore - Raw data used to build the charts"
        _stil_titlu(ws["A1"])

        ws.merge_cells("A2:N3")
        ws["A2"] = (
            "These values are used only to generate the charts from sheet 3. "
            "You can ignore this sheet during review."
        )
        _stil_nota(ws["A2"])
        ws.row_dimensions[2].height = 26
        ws.row_dimensions[3].height = 26

        ws["A5"] = "Fatigue Trend Data"
        _stil_titlu(ws["A5"])

        _stil_antet(ws.cell(6, 1, "Time"))
        for idx, student_id in enumerate(student_ids, start=2):
            _stil_antet(ws.cell(6, idx, f"Student {display_student_ids.get(str(student_id), idx - 1)}"))

        for row_idx, punct in enumerate(self.trend_buffer, start=7):
            ws.cell(row_idx, 1, punct["timestamp_label"]).border = _border_subtire()
            lookup = {
                str(item.get("student_id", "")).strip(): item
                for item in punct.get("studenti", [])
                if str(item.get("student_id", "")).strip() in student_id_set
            }
            for col_idx, student_id in enumerate(student_ids, start=2):
                cell = ws.cell(
                    row_idx,
                    col_idx,
                    None if lookup.get(student_id) is None else _rotunjeste(lookup[student_id].get("fatigue_pct", 0.0)),
                )
                cell.border = _border_subtire()
                if cell.value is not None:
                    cell.number_format = "0.00"

        att_title_row = len(self.trend_buffer) + 10
        ws.cell(att_title_row, 1, "Attention Trend Data")
        _stil_titlu(ws.cell(att_title_row, 1))

        _stil_antet(ws.cell(att_title_row + 1, 1, "Time"))
        for idx, student_id in enumerate(student_ids, start=2):
            _stil_antet(ws.cell(att_title_row + 1, idx, f"Student {display_student_ids.get(str(student_id), idx - 1)}"))

        for row_idx, punct in enumerate(self.trend_buffer, start=att_title_row + 2):
            ws.cell(row_idx, 1, punct["timestamp_label"]).border = _border_subtire()
            lookup = {
                str(item.get("student_id", "")).strip(): item
                for item in punct.get("studenti", [])
                if str(item.get("student_id", "")).strip() in student_id_set
            }
            for col_idx, student_id in enumerate(student_ids, start=2):
                cell = ws.cell(
                    row_idx,
                    col_idx,
                    None if lookup.get(student_id) is None else _rotunjeste(lookup[student_id].get("attention_pct", 0.0)),
                )
                cell.border = _border_subtire()
                if cell.value is not None:
                    cell.number_format = "0.00"

        _latime_automata(ws, min_width=14, max_width=22)
        return student_ids, timestamps, att_title_row + 1

    def _construieste_sheet_grafice(
        self,
        wb: Workbook,
        student_ids: List[str],
        timestamps: List[str],
        attention_header_row: int,
    ) -> None:
        ws = wb.create_sheet("Student Trends")
        ws.sheet_view.showGridLines = True

        ws.merge_cells("A1:N1")
        ws["A1"] = "Student Fatigue and Attention Trends"
        _stil_titlu(ws["A1"])

        ws.merge_cells("A2:N3")
        ws["A2"] = (
            "The charts below show how fatigue and attention changed during the session "
            "for the students included in the report."
        )
        _stil_nota(ws["A2"])
        ws.row_dimensions[2].height = 26
        ws.row_dimensions[3].height = 26

        if not student_ids or not timestamps:
            ws["A5"] = "No trend data available for this session."
            return

        ws_date = wb["Ignore - Trend Data"]

        chart_fatigue = LineChart()
        chart_fatigue.title = "Fatigue Over Time"
        chart_fatigue.y_axis.title = "Fatigue (%)"
        chart_fatigue.x_axis.title = "Time"
        chart_fatigue.height = 10
        chart_fatigue.width = 26
        chart_fatigue.style = 10
        chart_fatigue.legend.position = "t"

        try:
            chart_fatigue.y_axis.scaling.min = 0
            chart_fatigue.y_axis.scaling.max = 100
        except Exception:
            pass

        try:
            skip = max(1, len(timestamps) // 10)
            chart_fatigue.x_axis.tickLblSkip = skip
            chart_fatigue.x_axis.tickMarkSkip = skip
        except Exception:
            pass

        data_fatigue = Reference(
            ws_date,
            min_col=2,
            max_col=1 + len(student_ids),
            min_row=6,
            max_row=6 + len(timestamps),
        )
        cats_fatigue = Reference(ws_date, min_col=1, min_row=7, max_row=6 + len(timestamps))
        chart_fatigue.add_data(data_fatigue, titles_from_data=True)
        chart_fatigue.set_categories(cats_fatigue)

        for series in chart_fatigue.series:
            try:
                series.marker.symbol = "none"
                series.graphicalProperties.line.width = 28000
                series.smooth = False
            except Exception:
                pass

        ws.add_chart(chart_fatigue, "A5")

        chart_attention = LineChart()
        chart_attention.title = "Attention Over Time"
        chart_attention.y_axis.title = "Attention (%)"
        chart_attention.x_axis.title = "Time"
        chart_attention.height = 10
        chart_attention.width = 26
        chart_attention.style = 10
        chart_attention.legend.position = "t"

        try:
            chart_attention.y_axis.scaling.min = 0
            chart_attention.y_axis.scaling.max = 100
        except Exception:
            pass

        try:
            skip = max(1, len(timestamps) // 10)
            chart_attention.x_axis.tickLblSkip = skip
            chart_attention.x_axis.tickMarkSkip = skip
        except Exception:
            pass

        attention_first_data_row = attention_header_row + 1
        attention_last_data_row = attention_header_row + len(timestamps)

        data_attention = Reference(
            ws_date,
            min_col=2,
            max_col=1 + len(student_ids),
            min_row=attention_header_row,
            max_row=attention_last_data_row,
        )
        cats_attention = Reference(
            ws_date,
            min_col=1,
            min_row=attention_first_data_row,
            max_row=attention_last_data_row,
        )
        chart_attention.add_data(data_attention, titles_from_data=True)
        chart_attention.set_categories(cats_attention)

        for series in chart_attention.series:
            try:
                series.marker.symbol = "none"
                series.graphicalProperties.line.width = 28000
                series.smooth = False
            except Exception:
                pass

        ws.add_chart(chart_attention, "A28")

    def exporta_raport_sesiune_xlsx(
        self,
        session_id: Optional[int],
        session_summary: Dict[str, Any],
        session_students: List[Dict[str, Any]],
    ) -> str:
        wb = Workbook()

        valid_student_ids = self._student_ids_valizi_pentru_raport(session_students)
        studenti_filtrati = self._filtreaza_studenti_sumar(session_students, valid_student_ids)

        self._construieste_sheet_overview(wb, session_summary)
        self._construieste_sheet_studenti(wb, studenti_filtrati, valid_student_ids)
        student_ids, timestamps, attention_header_row = self._construieste_sheet_date_trend(
            wb,
            valid_student_ids,
        )
        self._construieste_sheet_grafice(wb, student_ids, timestamps, attention_header_row)

        desired_order = [
            "Session Overview",
            "Students Requiring Attention",
            "Student Trends",
            "Ignore - Trend Data",
        ]
        wb._sheets = [wb[name] for name in desired_order if name in wb.sheetnames]

        out_dir = Path(getattr(config, "SESSION_REPORTS_DIR", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)

        session_suffix = f"_{session_id}" if session_id is not None else ""
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"raport_sesiune{session_suffix}_{ts_now}.xlsx"

        wb.save(out_path)

        if session_id is not None:
            update_session_raport_path(self.db_path, session_id, str(out_path))

        return str(out_path)