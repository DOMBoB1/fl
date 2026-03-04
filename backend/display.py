import cv2

def resize_to_fill(frame, target_w, target_h):
    """
    Resize to fill window without distortion:
    keeps aspect ratio, center-crops excess, no black bars.
    """
    h, w = frame.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return frame

    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]
