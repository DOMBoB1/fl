import os
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "antre", "poz")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.csv")
OUTPUT_DIR = os.path.join(DATASET_DIR, "eval_output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "yolo_eval_results.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "yolo_eval_summary.csv")

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trash_code", "yolov8n.pt")
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
IMG_SIZE = 640


@dataclass
class Box:
    image_name: str
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: Optional[float] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_weights_path() -> str:
    candidates = [
        os.path.join(BASE_DIR, "best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "detect", "classroom_detector", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt"),
        WEIGHTS_PATH,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Missing weights file. Tried:\n- " + "\n- ".join(candidates)
    )


def norm_gt_label(label: str) -> Optional[str]:
    label = str(label).strip().lower()
    if label in {"face", "side_face"}:
        return "face"
    if label == "head":
        return "head"
    return None


def norm_pred_label(label: str) -> Optional[str]:
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"face", "side_face"}:
        return "face"
    if label == "head":
        return "head"
    return None


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_gt(csv_path: str) -> Dict[str, List[Box]]:
    grouped: Dict[str, List[Box]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = str(row["image_name"]).strip()
            cls = norm_gt_label(row["label"])
            if cls is None:
                continue

            box = Box(
                image_name=image_name,
                cls=cls,
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
            )
            grouped.setdefault(image_name, []).append(box)

    return grouped


def predict_image(model: YOLO, image_path: str) -> List[Box]:
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    preds: List[Box] = []
    names = model.names

    for result in results:
        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box_xyxy, cls_id, conf in zip(xyxy, cls_ids, confs):
            if isinstance(names, dict):
                cls_name = names.get(int(cls_id), str(cls_id))
            else:
                cls_name = names[int(cls_id)]

            cls = norm_pred_label(cls_name)
            if cls is None:
                continue

            preds.append(
                Box(
                    image_name=os.path.basename(image_path),
                    cls=cls,
                    x1=float(box_xyxy[0]),
                    y1=float(box_xyxy[1]),
                    x2=float(box_xyxy[2]),
                    y2=float(box_xyxy[3]),
                    conf=float(conf),
                )
            )

    return preds


def greedy_match(gt_boxes: List[Box], pred_boxes: List[Box], cls_name: str) -> Tuple[int, int, int]:
    gt_idx = [i for i, b in enumerate(gt_boxes) if b.cls == cls_name]
    pred_idx = [i for i, b in enumerate(pred_boxes) if b.cls == cls_name]

    pairs = []
    for i in gt_idx:
        for j in pred_idx:
            cur_iou = iou(
                (gt_boxes[i].x1, gt_boxes[i].y1, gt_boxes[i].x2, gt_boxes[i].y2),
                (pred_boxes[j].x1, pred_boxes[j].y1, pred_boxes[j].x2, pred_boxes[j].y2),
            )
            if cur_iou >= IOU_THRESHOLD:
                pairs.append((i, j, cur_iou))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_gt = set()
    used_pred = set()
    tp = 0

    for i, j, _ in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        tp += 1

    fp = len(pred_idx) - len(used_pred)
    fn = len(gt_idx) - len(used_gt)

    return tp, fp, fn


def evaluate() -> None:
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(f"Missing annotations file: {ANNOTATIONS_FILE}")

    resolved_weights_path = resolve_weights_path()

    gt = load_gt(ANNOTATIONS_FILE)
    model = YOLO(resolved_weights_path)

    totals = {
        "face": {"tp": 0, "fp": 0, "fn": 0},
        "head": {"tp": 0, "fp": 0, "fn": 0},
    }

    per_image = []
    missing_images = []
    processed = 0

    for image_name, gt_boxes in gt.items():
        image_path = os.path.join(DATASET_DIR, image_name)
        if not os.path.exists(image_path):
            missing_images.append(image_name)
            continue

        pred_boxes = predict_image(model, image_path)

        row = {"image_name": image_name}

        for cls_name in ("face", "head"):
            tp, fp, fn = greedy_match(gt_boxes, pred_boxes, cls_name)
            totals[cls_name]["tp"] += tp
            totals[cls_name]["fp"] += fp
            totals[cls_name]["fn"] += fn
            row[f"{cls_name}_tp"] = tp
            row[f"{cls_name}_fp"] = fp
            row[f"{cls_name}_fn"] = fn

        per_image.append(row)
        processed += 1

    results = {
        "config": {
            "dataset_dir": DATASET_DIR,
            "annotations_file": ANNOTATIONS_FILE,
            "weights_path": resolved_weights_path,
            "iou_threshold": IOU_THRESHOLD,
            "conf_threshold": CONF_THRESHOLD,
            "img_size": IMG_SIZE,
        },
        "dataset": {
            "images_with_gt": len(gt),
            "images_processed": processed,
            "missing_images": len(missing_images),
            "missing_image_names": missing_images,
        },
        "classes": {},
        "per_image": per_image,
    }

    summary_rows = []

    for cls_name in ("face", "head"):
        tp = totals[cls_name]["tp"]
        fp = totals[cls_name]["fp"]
        fn = totals[cls_name]["fn"]

        m = metrics(tp, fp, fn)
        acc_like = safe_div(tp, tp + fp + fn)

        results["classes"][cls_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy_like": acc_like,
            **m,
        }

        summary_rows.extend([
            {"metric": f"{cls_name}_accuracy_like", "value": acc_like},
            {"metric": f"{cls_name}_precision", "value": m["precision"]},
            {"metric": f"{cls_name}_recall", "value": m["recall"]},
            {"metric": f"{cls_name}_f1", "value": m["f1"]},
        ])

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Images with gt: {len(gt)}")
    print(f"Images processed: {processed}")
    print(f"Missing images: {len(missing_images)}")

    for cls_name in ("face", "head"):
        cls_res = results["classes"][cls_name]
        print(f"{cls_name} accuracy_like: {cls_res['accuracy_like'] * 100:.2f}%")
        print(f"{cls_name} precision: {cls_res['precision'] * 100:.2f}%")
        print(f"{cls_name} recall: {cls_res['recall'] * 100:.2f}%")
        print(f"{cls_name} f1: {cls_res['f1'] * 100:.2f}%")

    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate()