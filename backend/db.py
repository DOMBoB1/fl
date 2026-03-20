import sqlite3
from typing import List, Dict


def _connect(db_path: str):
    return sqlite3.connect(db_path)


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            interval_s INTEGER NOT NULL,
            faces_avg REAL NOT NULL,
            fatigue_avg REAL NOT NULL,
            attention_avg REAL NOT NULL,
            alert_count INTEGER NOT NULL,
            max_fatigue REAL NOT NULL,
            min_attention REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_snapshot(
    db_path: str,
    created_at: float,
    interval_s: int,
    faces_avg: float,
    fatigue_avg: float,
    attention_avg: float,
    alert_count: int,
    max_fatigue: float,
    min_attention: float,
) -> int:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO analysis_snapshots (
            created_at,
            interval_s,
            faces_avg,
            fatigue_avg,
            attention_avg,
            alert_count,
            max_fatigue,
            min_attention
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        interval_s,
        faces_avg,
        fatigue_avg,
        attention_avg,
        alert_count,
        max_fatigue,
        min_attention,
    ))

    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_recent_snapshots(db_path: str, limit: int = 3) -> List[Dict]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, created_at, interval_s, faces_avg, fatigue_avg,
               attention_avg, alert_count, max_fatigue, min_attention
        FROM analysis_snapshots
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_all_snapshots(db_path: str) -> List[Dict]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, created_at, interval_s, faces_avg, fatigue_avg,
               attention_avg, alert_count, max_fatigue, min_attention
        FROM analysis_snapshots
        ORDER BY created_at DESC
    """)

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows