# pylint: disable=no-member
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
from cv2 import resize, INTER_LINEAR

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


@dataclass
class MonitorStats:
    faces: int = 0
    class_avg_fatigue_pct: int = 0   # 0..100
    class_avg_attention_pct: int = 0
    alert_active: bool = False
    fps: float = 0.0


class ClassroomMonitorGUI:
    """
    GUI separat (Tkinter).
    Primește callbacks:
      - frame_provider() -> np.ndarray BGR (OpenCV)
      - stats_provider() -> MonitorStats sau dict compatibil
      - start_cb() / stop_cb()
    """

    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        stats_provider: Callable[[], MonitorStats],
        start_cb: Optional[Callable[[], None]] = None,
        stop_cb: Optional[Callable[[], None]] = None,
        title: str = "Classroom Monitor",
        target_fps: int = 15,
    ):
        self.frame_provider = frame_provider
        self.stats_provider = stats_provider
        self.start_cb = start_cb
        self.stop_cb = stop_cb
        self.title = title

        self.target_delay_ms = max(1, int(1000 / max(1, target_fps)))

        self._running = False
        self._last_frame_time = time.time()
        self._tk_img = None

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.minsize(1100, 650)

        self._build_layout()
        self._wire_events()

    # -------------------------
    # UI Layout
    # -------------------------

    def _build_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Video area
        self.video_frame = ttk.Frame(self.root, padding=8)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, background="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Sidebar
        self.side = ttk.Frame(self.root, padding=12)
        self.side.grid(row=0, column=1, sticky="nsew")
        self.side.columnconfigure(0, weight=1)

        # Controls
        controls = ttk.LabelFrame(self.side, text="Control", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.btn_start = ttk.Button(controls, text="Start", command=self.start)
        self.btn_stop = ttk.Button(controls, text="Stop", command=self.stop)
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Stats
        stats = ttk.LabelFrame(self.side, text="Statistici", padding=10)
        stats.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        stats.columnconfigure(0, weight=1)

        self.var_faces = tk.StringVar(value="0")
        self.var_fatigue = tk.StringVar(value="0%")
        self.var_attention = tk.StringVar(value="0%")
        self.var_alert = tk.StringVar(value="OFF")
        self.var_fps = tk.StringVar(value="0.0")

        self._row_label(stats, "Fețe:", self.var_faces, 0)
        self._row_label(stats, "Fatigue Avg:", self.var_fatigue, 1)
        self._row_label(stats, "Attention Avg:", self.var_attention, 2)
        self._row_label(stats, "Alertă:", self.var_alert, 3)
        self._row_label(stats, "FPS:", self.var_fps, 4)

        
        # Info (jos)
        helpbox = ttk.LabelFrame(self.side, text="Info", padding=10)
        helpbox.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        helpbox.columnconfigure(0, weight=1)

        txt = (
            "Info:\n"
            "  • Start/Stop din butoane\n"
            "  • ESC închide aplicația\n"
        )
        ttk.Label(helpbox, text=txt, justify="left").grid(row=0, column=0, sticky="w")

        # Optional style
        try:
            style = ttk.Style()
            style.theme_use("clam")
        except Exception:
            pass

    def _row_label(self, parent, name: str, var: tk.StringVar, row: int):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="ew", pady=2)
        frm.columnconfigure(1, weight=1)
        ttk.Label(frm, text=name).grid(row=0, column=0, sticky="w")
        ttk.Label(frm, textvariable=var).grid(row=0, column=1, sticky="e")

    # -------------------------
    # Fatigue color bar
    # -------------------------

    def _build_fatigue_scale(self, parent, row: int):
        # fără alb; un gri de UI
        c = tk.Canvas(parent, height=40, highlightthickness=0, bg="#ECECEC")
        c.grid(row=row, column=0, sticky="ew")
        c.bind("<Configure>", lambda e: self._draw_fatigue_scale(c))
        self._fatigue_scale_canvas = c
        self._draw_fatigue_scale(c)

    def _draw_fatigue_scale(self, c: tk.Canvas):
        c.delete("all")

        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())

        pad = 10
        x1 = pad
        x2 = w - pad

        bar_h = 16
        y = h // 2
        y1 = y - bar_h // 2
        y2 = y + bar_h // 2

        def x_at(pct: int) -> int:
            return int(x1 + (pct / 100.0) * (x2 - x1))

        # 3 segmente, ca în exemplu
        segments = [
            (0, 30,  "#E57373"),  # red
            (30, 70, "#FFD54F"),  # yellow
            (70, 100, "#81C784"), # green
        ]

        for a, b, color in segments:
            c.create_rectangle(x_at(a), y1, x_at(b), y2, fill=color, outline=color)

        # border subtil
        c.create_rectangle(x1, y1, x2, y2, outline="#2A2A2A", width=1)

        # markeri discreți (opțional, dar ajută)
        for m in (0, 30, 70, 100):
            xm = x_at(m)
            c.create_line(xm, y1 - 6, xm, y2 + 6, fill="#2A2A2A", width=1)

    # -------------------------
    # Events
    # -------------------------

    def _wire_events(self):
        self.root.bind("<Escape>", lambda e: self.close())

    # -------------------------
    # Public API
    # -------------------------

    def run(self):
        self._schedule_tick()
        self.root.mainloop()

    def start(self):
        if self._running:
            return
        self._running = True
        if self.start_cb:
            self.start_cb()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self.stop_cb:
            self.stop_cb()

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self.root.destroy()

    def get_alert_threshold_pct(self) -> int:
        return int(self.var_alert_threshold.get())

    # -------------------------
    # Tick / render
    # -------------------------

    def _schedule_tick(self):
        self.root.after(self.target_delay_ms, self._tick)

    def _tick(self):
        try:
            self._update_stats()
            if self._running:
                self._update_frame()
        except Exception:
            # nu omorâm UI-ul dintr-o excepție
            pass
        finally:
            self._schedule_tick()

    def _update_stats(self):
        s = self.stats_provider()
        if isinstance(s, dict):
            s = MonitorStats(**s)

        self.var_faces.set(str(s.faces))
        self.var_fatigue.set(f"{int(s.class_avg_fatigue_pct)}%")
        self.var_attention.set(f"{int(s.class_avg_attention_pct)}%")
        self.var_alert.set("ON" if s.alert_active else "OFF")
        self.var_fps.set(f"{s.fps:.1f}")

    def _update_frame(self):
        frame = self.frame_provider()
        if frame is None:
            return

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        rgb_fit = self._resize_to_fill(rgb, cw, ch)

        img = Image.fromarray(rgb_fit)
        self._tk_img = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _resize_to_fill(self, img_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized_img = resize(img_rgb, (new_w, new_h), interpolation=INTER_LINEAR)

        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        return resized_img[y0:y0 + target_h, x0:x0 + target_w]

    def _on_threshold_slider(self, _val):
        self.var_alert_threshold.set(int(float(self.slider_alert.get())))
        if hasattr(self, "lbl_thr") and self.lbl_thr is not None:
            self.lbl_thr.config(text=f"{self.var_alert_threshold.get()}%")
