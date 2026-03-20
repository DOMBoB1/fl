import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

DB_PATH = "classroom_monitor.db"
DATASET_ROOT = Path("dataset_sessions")


def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def session_folder_name(dt: datetime) -> str:
    return dt.strftime("session_%Y%m%d_%H%M%S")


def init_dataset_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            session_name TEXT NOT NULL,
            folder_path TEXT NOT NULL,
            source TEXT,
            notes TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            saved_at TEXT NOT NULL,
            frame_path TEXT NOT NULL,
            height INTEGER NOT NULL,
            width INTEGER NOT NULL,
            channels INTEGER NOT NULL,
            extra_json TEXT,
            FOREIGN KEY(session_id) REFERENCES dataset_sessions(id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_dataset_frames_session_id
        ON dataset_frames(session_id)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_dataset_frames_frame_index
        ON dataset_frames(frame_index)
    """)

    conn.commit()
    conn.close()


def start_dataset_session(
    source: str = "webcam",
    notes: str = "",
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    started_dt = datetime.utcnow()
    session_name = session_folder_name(started_dt)
    folder_path = DATASET_ROOT / session_name
    frames_path = folder_path / "frames"

    frames_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO dataset_sessions (started_at, session_name, folder_path, source, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (
        started_dt.strftime("%Y-%m-%d %H:%M:%S"),
        session_name,
        str(folder_path),
        source,
        notes
    ))

    session_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "session_id": session_id,
        "session_name": session_name,
        "folder_path": str(folder_path),
        "frames_path": str(frames_path),
    }


def stop_dataset_session(session_id: int, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        UPDATE dataset_sessions
        SET ended_at = ?
        WHERE id = ?
    """, (utc_now_str(), session_id))

    conn.commit()
    conn.close()


def save_frame_sample(
    session_id: int,
    session_folder: str,
    frame,
    frame_index: int,
    extra: Optional[Dict[str, Any]] = None,
    db_path: str = DB_PATH
) -> Optional[str]:
    if frame is None:
        return None

    if not isinstance(frame, np.ndarray):
        return None

    if frame.ndim == 2:
        h, w = frame.shape
        c = 1
    elif frame.ndim == 3:
        h, w, c = frame.shape
    else:
        return None

    frames_dir = Path(session_folder) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    filename = f"frame_{frame_index:08d}.npz"
    frame_path = frames_dir / filename

    np.savez_compressed(
        frame_path,
        frame=frame
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO dataset_frames (
            session_id,
            frame_index,
            saved_at,
            frame_path,
            height,
            width,
            channels,
            extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        frame_index,
        utc_now_str(),
        str(frame_path),
        int(h),
        int(w),
        int(c),
        json.dumps(extra, ensure_ascii=False) if extra is not None else None
    ))

    conn.commit()
    conn.close()

    return str(frame_path)


class FrameDatasetRecorder:
    def __init__(self, save_every_n_frames: int = 1):
        self.save_every_n_frames = max(1, int(save_every_n_frames))
        self.enabled = False
        self.session_id: Optional[int] = None
        self.session_folder: Optional[str] = None
        self.frame_counter = 0

    def start(self, source: str = "webcam", notes: str = "") -> None:
        init_dataset_db()
        info = start_dataset_session(source=source, notes=notes)
        self.session_id = info["session_id"]
        self.session_folder = info["folder_path"]
        self.frame_counter = 0
        self.enabled = True

    def stop(self) -> None:
        if self.session_id is not None:
            stop_dataset_session(self.session_id)

        self.enabled = False
        self.session_id = None
        self.session_folder = None
        self.frame_counter = 0

    def maybe_save(
        self,
        frame,
        extra: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if not self.enabled or self.session_id is None or self.session_folder is None:
            return None

        self.frame_counter += 1

        if self.frame_counter % self.save_every_n_frames != 0:
            return None

        return save_frame_sample(
            session_id=self.session_id,
            session_folder=self.session_folder,
            frame=frame,
            frame_index=self.frame_counter,
            extra=extra
        )