# =========================
# Compatibility aliases
# (so main.py can use consistent names)
# =========================

# Window name
if "WIN_NAME" not in globals():
    if "WINDOW_NAME" in globals():
        WIN_NAME = WINDOW_NAME
    else:
        WIN_NAME = "Classroom Monitor"
        WINDOW_NAME = WIN_NAME

# Tracking
if "TRACK_TTL_S" not in globals():
    # common older naming
    if "TRACK_TTL_SECONDS" in globals():
        TRACK_TTL_S = TRACK_TTL_SECONDS
    else:
        TRACK_TTL_S = 3.0

if "IOU_MATCH_T" not in globals():
    if "MATCH_IOU_THRESHOLD" in globals():
        IOU_MATCH_T = MATCH_IOU_THRESHOLD
    else:
        IOU_MATCH_T = 0.25

if "CENTER_DIST_T" not in globals():
    if "MATCH_CENTER_DIST_PX" in globals():
        CENTER_DIST_T = MATCH_CENTER_DIST_PX
    else:
        CENTER_DIST_T = 140

if "MIN_TRACK_POINTS" not in globals():
    MIN_TRACK_POINTS = 15

# Detection / crop
if "DETECT_EVERY_N_FRAMES" not in globals():
    DETECT_EVERY_N_FRAMES = 6

if "DETECT_WIDTH" not in globals():
    DETECT_WIDTH = 640

if "CROP_SIZE" not in globals():
    CROP_SIZE = 256

# Fatigue
if "EYES_CLOSED_EAR_T" not in globals():
    EYES_CLOSED_EAR_T = 0.22

if "DROWSY_CLOSED_MS" not in globals():
    DROWSY_CLOSED_MS = 900

if "PERCLOS_WINDOW_S" not in globals():
    PERCLOS_WINDOW_S = 60

# Attention (gaze)
if "LOOK_MIN" not in globals():
    LOOK_MIN = 0.32

if "LOOK_MAX" not in globals():
    LOOK_MAX = 0.68

# Alert
if "ALERT_CLASS_AVG_PCT" not in globals():
    ALERT_CLASS_AVG_PCT = 60

if "ALERT_HOLD_S" not in globals():
    ALERT_HOLD_S = 8
# =========================
# App / camera
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

WINDOW_NAME = "Multi-Person Fatigue & Attention Detection"

# Keep aspect ratio and fill window nicely (letterbox/pillarbox)
USE_LETTERBOX_VIEW = True


# =========================
# Multi-person tracking
# =========================
MAX_STUDENTS = 25

# Track matching thresholds (reduce "new IDs")
TRACK_TTL_SECONDS = 3.0        # keep IDs alive if briefly lost
MATCH_IOU_THRESHOLD = 0.20     # IoU jitter tolerance
MATCH_CENTER_DIST_PX = 120     # pixels; increase if camera is far/wide


# =========================
# FaceMesh
# =========================
ENABLE_FACE_DETECTION = True

# IMPORTANT: True enables iris landmarks (for gaze-based attention)
REFINE_LANDMARKS = True

MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5


# =========================
# Fatigue metrics
# =========================
# EAR threshold for "eyes closed"
EYE_AR_THRESHOLD = 0.22

# Blink threshold (can be lower than drowsiness threshold)
EAR_BLINK_THRESHOLD = 0.20
EAR_BLINK_CONSEC_FRAMES = 3

# Time windows
EAR_WINDOW_SECONDS = 10.0
BLINK_WINDOW_SECONDS = 60.0

# How many samples before we trust metrics
METRIC_READY_MIN_FRAMES = 20

# Score update interval (for average score display)
SCORE_UPDATE_INTERVAL = 2.0
AVERAGE_ACTIVE_WINDOW = 10.0

# Weights for fatigue score (0–100)
EAR_WEIGHT = 0.6
BLINK_WEIGHT = 0.4


# =========================
# Attention (gaze)
# =========================
# Gaze ratio in [0..1]; center ~ 0.5
GAZE_CENTER_MIN = 0.35
GAZE_CENTER_MAX = 0.65

ATTENTION_WINDOW_SECONDS = 10.0   # attention % computed over last N seconds


# =========================
# Visualization
# =========================
SHOW_FACE_BOX = True
SHOW_DEBUG_TEXT = False

# Logging
LOGGING_ENABLED = True
