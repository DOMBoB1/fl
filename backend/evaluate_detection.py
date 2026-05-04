import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2

from engine import MonitoringEngine


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = (BASE_DIR / ".." / "antre" / "poz").resolve()
CSV_PATH = IMAGES_DIR / "annotations.csv"

EVAL_OUTPUT_DIR = BASE_DIR / "eval_output"
REPORTS_DIR = BASE_DIR / "reports"

WARMUP_FRAMES_PER_IMAGE = 5


def resolve_image_path(img_name: str) -> Path:
    p = Path(str(img_name).strip())

    if p.is_absolute() and p.exists():
        return p

    candidate = IMAGES_DIR / p.name
    if candidate.exists():
        return candidate

    candidate = IMAGES_DIR / p
    if candidate.exists():
        return candidate

    return IMAGES_DIR / p.name


def load_ground_truth_counts(csv_path: Path):
    image_counts = defaultdict(lambda: {"faces": 0, "heads": 0})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            img_name = (
                row.get("image_name")
                or row.get("image_path")
                or row.get("filename")
                or row.get("image")
            )

            if not img_name:
                continue

            label = str(row.get("label", "")).strip().lower()
            img_key = Path(str(img_name)).name

            if label in {"head", "heads", "cap", "head_box"}:
                image_counts[img_key]["heads"] += 1
            else:
                image_counts[img_key]["faces"] += 1

    for img_key, counts in image_counts.items():
        if counts["heads"] == 0:
            counts["heads"] = counts["faces"]

    return image_counts


def safe_pct(value: float) -> float:
    return round(float(value) * 100.0, 2)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gt_counts = load_ground_truth_counts(CSV_PATH)

    if not gt_counts:
        raise ValueError("No valid labels found in annotations.csv")

    engine = MonitoringEngine()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_abs_error = 0
    total_images = 0

    details = []

    for img_name, gt in gt_counts.items():
        img_path = resolve_image_path(img_name)

        if not img_path.exists():
            print(f"Missing image: {img_path}")
            continue

        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"Could not read image: {img_path}")
            continue

        engine._reset_live_runtime_state()

        stats = None
        for _ in range(WARMUP_FRAMES_PER_IMAGE):
            stats = engine.process_external_frame(frame, want_boxes=False)

        if not isinstance(stats, dict):
            print(f"Invalid engine output for: {img_name}")
            continue

        true_faces = int(gt["faces"])
        pred_faces = int(stats.get("faces", 0))

        tp = min(pred_faces, true_faces)
        fp = max(pred_faces - true_faces, 0)
        fn = max(true_faces - pred_faces, 0)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        abs_error = abs(pred_faces - true_faces)
        total_abs_error += abs_error
        total_images += 1

        image_accuracy = 1.0 - (abs_error / max(true_faces, 1))
        image_accuracy = max(0.0, min(1.0, image_accuracy))

        details.append(
            {
                "image": img_name,
                "true_faces": true_faces,
                "predicted_faces": pred_faces,
                "absolute_error": abs_error,
                "accuracy_pct": safe_pct(image_accuracy),
            }
        )

        print(
            f"{img_name}: true={true_faces}, predicted={pred_faces}, "
            f"accuracy={safe_pct(image_accuracy)}%"
        )

    if total_images == 0:
        raise ValueError("No images were evaluated")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    accuracy = sum(x["accuracy_pct"] for x in details) / len(details)

    results = {
        "images_evaluated": total_images,
        "accuracy_pct": round(accuracy, 2),
        "precision_pct": safe_pct(precision),
        "recall_pct": safe_pct(recall),
        "f1_pct": safe_pct(f1),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "mean_absolute_error": round(total_abs_error / total_images, 2),
        "details": details,
    }

    metrics_path_1 = EVAL_OUTPUT_DIR / "metrics.json"
    metrics_path_2 = REPORTS_DIR / "last_eval_metrics.json"

    with open(metrics_path_1, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open(metrics_path_2, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nDONE")
    print(json.dumps(results, indent=4))
    print(f"\nSaved: {metrics_path_1}")
    print(f"Saved: {metrics_path_2}")


if __name__ == "__main__":
    main()