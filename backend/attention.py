import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

# FaceMesh indices
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]


def lm_xy(lms, idx, w, h):
    p = lms.landmark[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def iris_center(lms, iris_ids, w, h):
    pts = [lm_xy(lms, i, w, h) for i in iris_ids]
    return np.mean(pts, axis=0)


def gaze_ratio_one_eye(lms, corner_ids, iris_ids, w, h):
    left_corner = lm_xy(lms, corner_ids[0], w, h)
    right_corner = lm_xy(lms, corner_ids[1], w, h)
    pupil = iris_center(lms, iris_ids, w, h)

    eye_width = np.linalg.norm(right_corner - left_corner)
    if eye_width < 1e-6:
        return 0.5

    ratio = (pupil[0] - left_corner[0]) / eye_width
    return float(np.clip(ratio, 0.0, 1.0))


def gaze_ratio(lms, w, h):
    gl = gaze_ratio_one_eye(lms, LEFT_EYE_CORNERS, LEFT_IRIS, w, h)
    gr = gaze_ratio_one_eye(lms, RIGHT_EYE_CORNERS, RIGHT_IRIS, w, h)
    return float((gl + gr) / 2.0)


def attentive_from_gaze(gaze, look_min=0.32, look_max=0.68, margin=0.08):
    g = float(np.clip(gaze, 0.0, 1.0))

    inner_min = look_min
    inner_max = look_max
    outer_min = max(0.0, look_min - margin)
    outer_max = min(1.0, look_max + margin)

    if inner_min <= g <= inner_max:
        return 100

    if g < inner_min:
        if g <= outer_min:
            return 0
        return int(100 * (g - outer_min) / (inner_min - outer_min + 1e-9))

    if g >= outer_max:
        return 0

    return int(100 * (outer_max - g) / (outer_max - inner_max + 1e-9))


def attentive_from_reference(
    gaze: float,
    reference: float,
    inner_tolerance: float = 0.08,
    outer_tolerance: float = 0.18,
) -> int:
    """
    Returns an attention score based on distance to a reference gaze direction.
    reference ~= dominant classroom gaze direction.
    """
    g = float(np.clip(gaze, 0.0, 1.0))
    ref = float(np.clip(reference, 0.0, 1.0))

    dist = abs(g - ref)

    if dist <= inner_tolerance:
        return 100

    if dist >= outer_tolerance:
        return 0

    score = 100.0 * (outer_tolerance - dist) / (outer_tolerance - inner_tolerance + 1e-9)
    return int(np.clip(round(score), 0, 100))


@dataclass
class AdaptiveAttentionConfig:
    enabled: bool = True

    # How long we observe the class to infer a dominant direction
    window_seconds: float = 300.0  # 5 minutes

    # We only trust the adaptive baseline if we saw enough valid class observations
    min_samples: int = 120

    # At least this many students should contribute in a frame to consider it representative
    min_students_per_frame: int = 2

    # If the classroom gaze varies too much, we do not adapt
    max_std_for_lock: float = 0.09

    # Start from the classic "look ahead" assumption
    default_reference: float = 0.50

    # Scoring against the current reference
    inner_tolerance: float = 0.08
    outer_tolerance: float = 0.18

    # Blend new stable dominant direction slowly
    ema_alpha: float = 0.08

    # Keep old behavior as fallback if adaptation is not reliable
    fallback_look_min: float = 0.32
    fallback_look_max: float = 0.68
    fallback_margin: float = 0.08


@dataclass
class AdaptiveAttentionState:
    config: AdaptiveAttentionConfig = field(default_factory=AdaptiveAttentionConfig)
    history: deque = field(default_factory=deque)
    current_reference: float = 0.50
    locked_reference: Optional[float] = None
    last_update_ts: float = field(default_factory=time.time)

    def __post_init__(self):
        self.current_reference = self.config.default_reference

    def reset(self):
        self.history.clear()
        self.current_reference = self.config.default_reference
        self.locked_reference = None
        self.last_update_ts = time.time()

    def _prune(self, now: float):
        cutoff = now - self.config.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def update_classroom_reference(
        self,
        gaze_values: Iterable[float],
        now: Optional[float] = None,
    ) -> float:
        """
        Update the dominant classroom gaze direction using the median gaze ratio
        from the current frame, if enough students are present.
        """
        if now is None:
            now = time.time()

        gaze_values = [
            float(np.clip(g, 0.0, 1.0))
            for g in gaze_values
            if g is not None and np.isfinite(g)
        ]

        if len(gaze_values) >= self.config.min_students_per_frame:
            frame_median = float(np.median(gaze_values))
            frame_std = float(np.std(gaze_values))
            self.history.append((now, frame_median, frame_std))

        self._prune(now)

        if not self.config.enabled:
            self.current_reference = self.config.default_reference
            self.locked_reference = None
            return self.current_reference

        if len(self.history) < self.config.min_samples:
            self.locked_reference = None
            return self.current_reference

        medians = np.array([x[1] for x in self.history], dtype=np.float32)
        stds = np.array([x[2] for x in self.history], dtype=np.float32)

        dominant = float(np.median(medians))
        spread = float(np.std(medians))
        avg_frame_std = float(np.mean(stds)) if len(stds) > 0 else 0.0

        # We adapt only if the class direction is reasonably stable
        stable_enough = spread <= self.config.max_std_for_lock and avg_frame_std <= self.config.max_std_for_lock

        if stable_enough:
            self.locked_reference = dominant
            self.current_reference = (
                (1.0 - self.config.ema_alpha) * self.current_reference
                + self.config.ema_alpha * dominant
            )
        else:
            self.locked_reference = None

        self.current_reference = float(np.clip(self.current_reference, 0.0, 1.0))
        return self.current_reference

    def score_attention(self, gaze: float) -> int:
        """
        Score one student's attention against the adaptive classroom reference
        when available; otherwise fallback to the old fixed-range logic.
        """
        g = float(np.clip(gaze, 0.0, 1.0))

        if self.config.enabled and self.locked_reference is not None:
            return attentive_from_reference(
                g,
                reference=self.current_reference,
                inner_tolerance=self.config.inner_tolerance,
                outer_tolerance=self.config.outer_tolerance,
            )

        return attentive_from_gaze(
            g,
            look_min=self.config.fallback_look_min,
            look_max=self.config.fallback_look_max,
            margin=self.config.fallback_margin,
        )

    def get_reference_info(self) -> dict:
        return {
            "adaptive_enabled": self.config.enabled,
            "reference_gaze": float(self.current_reference),
            "reference_locked": self.locked_reference is not None,
            "history_size": len(self.history),
            "window_seconds": float(self.config.window_seconds),
        }


def update_attention_for_class(
    gaze_values: List[float],
    adaptive_state: AdaptiveAttentionState,
    now: Optional[float] = None,
) -> Tuple[List[int], dict]:
    """
    Helper for a whole frame:
    - updates classroom dominant direction
    - returns attention score for each gaze value
    - returns debug info for UI / logs
    """
    reference = adaptive_state.update_classroom_reference(gaze_values, now=now)
    scores = [adaptive_state.score_attention(g) for g in gaze_values]

    info = adaptive_state.get_reference_info()
    info["frame_reference"] = reference
    info["frame_students"] = len(gaze_values)

    return scores, info