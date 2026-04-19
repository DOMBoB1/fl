from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_TO_ID = {
    "face": 0,
    "head": 1,
    "side_face": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO train/val dataset from antre/poz/annotations.csv"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="../antre/poz/annotations.csv",
        help="Path to annotations.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../antre/poz/yolo_dataset",
        help="Output YOLO dataset directory",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (0..1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.resolve()
    return (Path(__file__).resolve().parent / candidate).resolve()


def _to_yolo_row(
    img_w: int, img_h: int, class_id: int, x1: float, y1: float, x2: float, y2: float
) -> str:
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    args = parse_args()
    annotations_path = _resolve(args.annotations)
    output_root = _resolve(args.output)

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.csv not found: {annotations_path}")

    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1")

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with annotations_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "image_name",
            "image_path",
            "image_width",
            "image_height",
            "label",
            "x1",
            "y1",
            "x2",
            "y2",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"annotations.csv missing columns: {sorted(missing)}")
        for row in reader:
            rows_by_image[row["image_name"]].append(row)

    image_names = sorted(rows_by_image.keys())
    if not image_names:
        raise ValueError("No annotation rows found.")

    rng = random.Random(args.seed)
    rng.shuffle(image_names)

    val_count = max(1, int(round(len(image_names) * args.val_ratio)))
    if val_count >= len(image_names):
        val_count = len(image_names) - 1
    val_set = set(image_names[:val_count])

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"

    for p in (images_train, images_val, labels_train, labels_val):
        p.mkdir(parents=True, exist_ok=True)
        for item in p.glob("*"):
            if item.is_file():
                item.unlink()

    split_rows: list[tuple[str, str]] = []
    for image_name in image_names:
        rows = rows_by_image[image_name]
        image_path = Path(rows[0]["image_path"])
        if not image_path.exists():
            print(f"[WARN] missing image on disk, skipping: {image_path}")
            continue

        split = "val" if image_name in val_set else "train"
        out_img = (images_val if split == "val" else images_train) / image_name
        out_label = (labels_val if split == "val" else labels_train) / (
            Path(image_name).stem + ".txt"
        )

        yolo_lines: list[str] = []
        img_w = int(float(rows[0]["image_width"]))
        img_h = int(float(rows[0]["image_height"]))
        for row in rows:
            label = row["label"].strip()
            if label not in CLASS_TO_ID:
                print(f"[WARN] unknown label '{label}' in {image_name}; skipped row")
                continue
            yolo_lines.append(
                _to_yolo_row(
                    img_w=img_w,
                    img_h=img_h,
                    class_id=CLASS_TO_ID[label],
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            )

        if not yolo_lines:
            print(f"[WARN] no valid labels for {image_name}; skipping file")
            continue

        shutil.copy2(image_path, out_img)
        out_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        split_rows.append((image_name, split))

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: face",
                "  1: head",
                "  2: side_face",
                "",
            ]
        ),
        encoding="utf-8",
    )

    split_summary = output_root / "split_summary.csv"
    with split_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "split"])
        writer.writerows(split_rows)

    train_n = sum(1 for _, s in split_rows if s == "train")
    val_n = sum(1 for _, s in split_rows if s == "val")
    print(f"[OK] Dataset rebuilt at: {output_root}")
    print(f"[OK] train images: {train_n}")
    print(f"[OK] val images: {val_n}")
    print(f"[OK] data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()

