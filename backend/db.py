import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _utc_text(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = _utc_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _real2(value: Any) -> float:
    try:
        return round(float(value), 2)
    except Exception:
        return 0.0


def _int0(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            started_at REAL NOT NULL,
            started_at_text TEXT NOT NULL,

            ended_at REAL,
            ended_at_text TEXT,

            source TEXT DEFAULT 'webcam',
            notes TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running',

            faces_avg REAL NOT NULL DEFAULT 0,
            heads_avg REAL NOT NULL DEFAULT 0,
            active_students_avg REAL NOT NULL DEFAULT 0,
            fatigue_avg REAL NOT NULL DEFAULT 0,
            attention_avg REAL NOT NULL DEFAULT 0,

            class_alert_type TEXT NOT NULL DEFAULT 'none',
            class_alert_level TEXT NOT NULL DEFAULT 'none',

            reason TEXT DEFAULT '',
            decision TEXT DEFAULT '',

            report_path TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id INTEGER NOT NULL,

            created_at REAL NOT NULL,
            created_at_text TEXT NOT NULL,

            faces_avg REAL NOT NULL DEFAULT 0,
            heads_avg REAL NOT NULL DEFAULT 0,
            active_students_avg REAL NOT NULL DEFAULT 0,

            fatigue_avg REAL NOT NULL DEFAULT 0,
            attention_avg REAL NOT NULL DEFAULT 0,

            fatigue_max REAL NOT NULL DEFAULT 0,
            attention_min REAL NOT NULL DEFAULT 0,

            class_alert_type TEXT NOT NULL DEFAULT 'none',
            class_alert_level TEXT NOT NULL DEFAULT 'none',

            student_alert_count INTEGER NOT NULL DEFAULT 0,
            critical_student_count INTEGER NOT NULL DEFAULT 0,

            reason TEXT DEFAULT '',
            decision TEXT DEFAULT '',

            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,

            fatigue_avg REAL NOT NULL DEFAULT 0,
            attention_avg REAL NOT NULL DEFAULT 0,

            fatigue_max REAL NOT NULL DEFAULT 0,
            attention_min REAL NOT NULL DEFAULT 0,

            final_alert_type TEXT NOT NULL DEFAULT 'none',
            final_severity TEXT NOT NULL DEFAULT 'none',

            reason TEXT DEFAULT '',
            decision TEXT DEFAULT '',

            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            UNIQUE(session_id, student_id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_session_id
        ON analysis_snapshots(session_id)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_created_at
        ON analysis_snapshots(created_at DESC)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_students_session_id
        ON session_students(session_id)
    """)

    conn.commit()
    conn.close()



def start_session(
    db_path: str,
    source: str = "webcam",
    notes: str = ""
) -> int:
    ts = _utc_ts()
    text = _utc_text(ts)

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO sessions (
            started_at,
            started_at_text,
            source,
            notes,
            status
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        ts,
        text,
        str(source or "webcam"),
        str(notes or ""),
        "running"
    ))

    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def finish_session(
    db_path: str,
    session_id: int,
    faces_avg: float = 0,
    heads_avg: float = 0,
    active_students_avg: float = 0,
    fatigue_avg: float = 0,
    attention_avg: float = 0,
    class_alert_type: str = "none",
    class_alert_level: str = "none",
    reason: str = "",
    decision: str = "",
    report_path: str = "",
    status: str = "finished",
) -> None:
    ts = _utc_ts()
    text = _utc_text(ts)

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        UPDATE sessions
        SET
            ended_at = ?,
            ended_at_text = ?,
            status = ?,

            faces_avg = ?,
            heads_avg = ?,
            active_students_avg = ?,
            fatigue_avg = ?,
            attention_avg = ?,

            class_alert_type = ?,
            class_alert_level = ?,
            reason = ?,
            decision = ?,
            report_path = ?
        WHERE id = ?
    """, (
        ts,
        text,
        str(status or "finished"),

        _real2(faces_avg),
        _real2(heads_avg),
        _real2(active_students_avg),
        _real2(fatigue_avg),
        _real2(attention_avg),

        str(class_alert_type or "none"),
        str(class_alert_level or "none"),
        str(reason or ""),
        str(decision or ""),
        str(report_path or ""),
        session_id
    ))

    conn.commit()
    conn.close()


def update_session_report_path(
    db_path: str,
    session_id: int,
    report_path: str
) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        UPDATE sessions
        SET report_path = ?
        WHERE id = ?
    """, (str(report_path or ""), session_id))

    conn.commit()
    conn.close()


def get_session(db_path: str, session_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM sessions
        WHERE id = ?
    """, (session_id,))

    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_session(db_path: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM sessions
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_sessions(db_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM sessions
        ORDER BY id DESC
        LIMIT ?
    """, (_int0(limit) if _int0(limit) > 0 else 20,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows



def insert_snapshot(
    db_path: str,
    session_id: int,
    created_at: Optional[float] = None,
    faces_avg: float = 0,
    heads_avg: float = 0,
    active_students_avg: float = 0,
    fatigue_avg: float = 0,
    attention_avg: float = 0,
    fatigue_max: float = 0,
    attention_min: float = 0,
    class_alert_type: str = "none",
    class_alert_level: str = "none",
    student_alert_count: int = 0,
    critical_student_count: int = 0,
    reason: str = "",
    decision: str = "",
) -> int:
    ts = float(created_at) if created_at is not None else _utc_ts()
    text = _utc_text(ts)

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO analysis_snapshots (
            session_id,
            created_at,
            created_at_text,

            faces_avg,
            heads_avg,
            active_students_avg,

            fatigue_avg,
            attention_avg,

            fatigue_max,
            attention_min,

            class_alert_type,
            class_alert_level,

            student_alert_count,
            critical_student_count,

            reason,
            decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        ts,
        text,

        _real2(faces_avg),
        _real2(heads_avg),
        _real2(active_students_avg),

        _real2(fatigue_avg),
        _real2(attention_avg),

        _real2(fatigue_max),
        _real2(attention_min),

        str(class_alert_type or "none"),
        str(class_alert_level or "none"),

        _int0(student_alert_count),
        _int0(critical_student_count),

        str(reason or ""),
        str(decision or ""),
    ))

    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_recent_snapshots(
    db_path: str,
    session_id: Optional[int] = None,
    limit: int = 3
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    safe_limit = _int0(limit) if _int0(limit) > 0 else 3

    if session_id is None:
        cur.execute("""
            SELECT *
            FROM analysis_snapshots
            ORDER BY created_at DESC
            LIMIT ?
        """, (safe_limit,))
    else:
        cur.execute("""
            SELECT *
            FROM analysis_snapshots
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, safe_limit))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_all_snapshots(
    db_path: str,
    session_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if session_id is None:
        cur.execute("""
            SELECT *
            FROM analysis_snapshots
            ORDER BY created_at ASC
        """)
    else:
        cur.execute("""
            SELECT *
            FROM analysis_snapshots
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows



def upsert_session_student(
    db_path: str,
    session_id: int,
    student_id: Any,
    fatigue_avg: float = 0,
    attention_avg: float = 0,
    fatigue_max: float = 0,
    attention_min: float = 0,
    final_alert_type: str = "none",
    final_severity: str = "none",
    reason: str = "",
    decision: str = "",
) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO session_students (
            session_id,
            student_id,
            fatigue_avg,
            attention_avg,
            fatigue_max,
            attention_min,
            final_alert_type,
            final_severity,
            reason,
            decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, student_id)
        DO UPDATE SET
            fatigue_avg = excluded.fatigue_avg,
            attention_avg = excluded.attention_avg,
            fatigue_max = excluded.fatigue_max,
            attention_min = excluded.attention_min,
            final_alert_type = excluded.final_alert_type,
            final_severity = excluded.final_severity,
            reason = excluded.reason,
            decision = excluded.decision
    """, (
        session_id,
        str(student_id),
        _real2(fatigue_avg),
        _real2(attention_avg),
        _real2(fatigue_max),
        _real2(attention_min),
        str(final_alert_type or "none"),
        str(final_severity or "none"),
        str(reason or ""),
        str(decision or ""),
    ))

    conn.commit()
    conn.close()


def replace_session_students(
    db_path: str,
    session_id: int,
    students: Iterable[Dict[str, Any]]
) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM session_students
        WHERE session_id = ?
    """, (session_id,))

    for student in students:
        cur.execute("""
            INSERT INTO session_students (
                session_id,
                student_id,
                fatigue_avg,
                attention_avg,
                fatigue_max,
                attention_min,
                final_alert_type,
                final_severity,
                reason,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            str(student.get("student_id", "")),
            _real2(student.get("fatigue_avg", 0)),
            _real2(student.get("attention_avg", 0)),
            _real2(student.get("fatigue_max", 0)),
            _real2(student.get("attention_min", 0)),
            str(student.get("final_alert_type", "none") or "none"),
            str(student.get("final_severity", "none") or "none"),
            str(student.get("reason", "") or ""),
            str(student.get("decision", "") or ""),
        ))

    conn.commit()
    conn.close()


def get_session_students(
    db_path: str,
    session_id: int,
    only_flagged: bool = False
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if only_flagged:
        cur.execute("""
            SELECT *
            FROM session_students
            WHERE session_id = ?
              AND final_severity != 'none'
            ORDER BY
                CASE final_severity
                    WHEN 'critical' THEN 0
                    WHEN 'warning' THEN 1
                    ELSE 2
                END,
                attention_min ASC,
                fatigue_max DESC
        """, (session_id,))
    else:
        cur.execute("""
            SELECT *
            FROM session_students
            WHERE session_id = ?
            ORDER BY
                CASE final_severity
                    WHEN 'critical' THEN 0
                    WHEN 'warning' THEN 1
                    ELSE 2
                END,
                attention_min ASC,
                fatigue_max DESC
        """, (session_id,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
