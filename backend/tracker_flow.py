import math
import numpy as np
import cv2

# pylint: disable=no-member  # OpenCV's cv2 bindings lack static attribute info

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def center_dist(a, b) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return float(math.hypot(ax - bx, ay - by))

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)

def sample_points_in_box(box, n=64):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 10 or h <= 10:
        return np.zeros((0, 1, 2), dtype=np.float32)

    cols = int(math.sqrt(n))
    rows = cols
    xs = np.linspace(x1 + w * 0.2, x2 - w * 0.2, cols)
    ys = np.linspace(y1 + h * 0.2, y2 - h * 0.2, rows)
    pts = np.array(  # pylint: disable=too-many-function-args
        [(x, y) for y in ys for x in xs], dtype=np.float32
    ).reshape(-1, 1, 2)
    return pts

def shift_box_by_flow(box, prev_pts, next_pts):
    if prev_pts is None or next_pts is None or len(prev_pts) == 0 or len(next_pts) == 0:
        return box, 0
    flow = (next_pts - prev_pts).reshape(-1, 2)
    dx = float(np.median(flow[:, 0]))
    dy = float(np.median(flow[:, 1]))
    x1, y1, x2, y2 = box
    return (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)), len(flow)

class FlowMultiTracker:
    """
    Stable multi-object tracking using Lucas–Kanade optical flow + periodic detection association.
    """
    def __init__(self, ttl_s=3.0, iou_t=0.25, dist_t=140, min_pts=15):
        self.ttl_s = ttl_s
        self.iou_t = iou_t
        self.dist_t = dist_t
        self.min_pts = min_pts

        self.tracks = {}  # tid -> dict(bbox, last_seen, pts)
        self.next_id = 1
        self.prev_gray = None

    def update_optical_flow(self, frame_bgr, now):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1) First frame: initialize
        if self.prev_gray is None:
            self.prev_gray = gray
            return

        # 2) If resolution changed, reset flow state (LK requires same size)
        if self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            for tid, tr in list(self.tracks.items()):
                tr["pts"] = sample_points_in_box(tr["bbox"], n=64)
                tr["last_seen"] = now
            return

        # 3) Normal LK update
        for tid, tr in list(self.tracks.items()):
            bbox = tr["bbox"]
            pts = tr.get("pts", None)

            if pts is None or len(pts) == 0:
                tr["pts"] = sample_points_in_box(bbox, n=64)
                continue

            nxt, st, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )

            if nxt is None or st is None:
                tr["pts"] = sample_points_in_box(bbox, n=64)
                continue

            st = st.flatten().astype(bool)
            good_prev = pts[st]
            good_next = nxt[st]

            if len(good_prev) < self.min_pts:
                tr["pts"] = sample_points_in_box(bbox, n=64)
                continue

            shifted, _n = shift_box_by_flow(
                bbox,
                good_prev.reshape(-1, 1, 2),
                good_next.reshape(-1, 1, 2)
            )

            tr["bbox"] = shifted
            tr["last_seen"] = now
            tr["pts"] = good_next.reshape(-1, 1, 2)

        self.prev_gray = gray

    def _associate(self, dets):
        tids = list(self.tracks.keys())
        unmatched_t = set(tids)
        unmatched_d = set(range(len(dets)))
        matches = []

        candidates = []
        for tid in tids:
            tb = self.tracks[tid]["bbox"]
            for di, db in enumerate(dets):
                ii = iou(tb, db)
                dd = center_dist(tb, db)
                if ii >= self.iou_t or dd <= self.dist_t:
                    score = ii - (dd / 1000.0)
                    candidates.append((score, tid, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        used_t, used_d = set(), set()
        for score, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            matches.append((tid, di))
            used_t.add(tid)
            used_d.add(di)
            unmatched_t.discard(tid)
            unmatched_d.discard(di)

        return matches, list(unmatched_t), list(unmatched_d)

    def sync_with_detections(self, frame_bgr, dets, now):
        H, W = frame_bgr.shape[:2]

        # clamp all dets first
        dets = [clamp_box(d, W, H) for d in dets]

        matches, _unmatched_t, unmatched_d = self._associate(dets)

        # refresh matched
        for tid, di in matches:
            self.tracks[tid]["bbox"] = dets[di]
            self.tracks[tid]["last_seen"] = now
            self.tracks[tid]["pts"] = sample_points_in_box(dets[di], n=64)

        # create new
        for di in unmatched_d:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": dets[di],
                "last_seen": now,
                "pts": sample_points_in_box(dets[di], n=64)
            }

    def drop_stale(self, now):
        for tid in list(self.tracks.keys()):
            if (now - self.tracks[tid]["last_seen"]) > self.ttl_s:
                del self.tracks[tid]

    def get_tracks(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        out = {}
        for tid, tr in self.tracks.items():
            out[tid] = clamp_box(tr["bbox"], W, H)
        return out
