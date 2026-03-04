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
    return (gl + gr) / 2.0

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
