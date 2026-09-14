#!/usr/bin/env python3
"""Build whole-building/partial-or-background Chinese-painting crops."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


BUILDING_CLASS = 1


def yolo_box(
    row: str, width: int, height: int, expansion: float = 0.08
) -> tuple[int, int, int, int]:
    _, cx, cy, bw, bh = map(float, row.split())
    bw *= 1.0 + 2.0 * expansion
    bh *= 1.0 + 2.0 * expansion
    x1 = max(0, round((cx - bw / 2) * width))
    y1 = max(0, round((cy - bh / 2) * height))
    x2 = min(width, round((cx + bw / 2) * width))
    y2 = min(height, round((cy + bh / 2) * height))
    return x1, y1, x2, y2


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def save_crop(image: Image.Image, box: tuple[int, int, int, int], path: Path) -> bool:
    if box[2] - box[0] < 12 or box[3] - box[1] < 12:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).convert("RGB").save(path, quality=94)
    return True


def partial_building_boxes(
    box: tuple[int, int, int, int],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Derive generic hard negatives from a labeled complete building box."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return [
        ("roof_only", (x1, y1, x2, round(y1 + 0.48 * height))),
        ("eave_band", (x1, round(y1 + 0.18 * height), x2, round(y1 + 0.62 * height))),
        ("left_fragment", (x1, y1, round(x1 + 0.54 * width), y2)),
        ("right_fragment", (round(x1 + 0.46 * width), y1, x2, y2)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(".dataset_cache/datasets/edmundxu/chineselandscapeobjectdetection/versions/4"),
    )
    parser.add_argument("--detector-data", type=Path, default=Path("training_data/building"))
    parser.add_argument("--output", type=Path, default=Path("training_data/building_classifier"))
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_by_name = {
        image.name: split
        for split in ("train", "val")
        for image in args.detector_data.joinpath("images", split).glob("*.jpg")
    }
    negative_candidates: dict[str, list[tuple[Path, tuple[int, int, int, int], str]]] = {
        "train": [],
        "val": [],
    }
    counts = {
        "train": {"building": 0, "not_building": 0},
        "val": {"building": 0, "not_building": 0},
    }

    for image_path in sorted(args.source.joinpath("images").glob("*.jpg")):
        split = split_by_name.get(image_path.name)
        if split is None:
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        rows = (
            args.source.joinpath("labels", f"{image_path.stem}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        building_rows = [row for row in rows if row.split()[0] == str(BUILDING_CLASS)]
        building_boxes = [yolo_box(row, width, height, 0.02) for row in building_rows]

        for index, row in enumerate(building_rows):
            box = yolo_box(row, width, height, 0.10)
            destination = args.output / split / "building" / f"{image_path.stem}_b{index:02d}.jpg"
            counts[split]["building"] += int(save_crop(image, box, destination))
            context_box = yolo_box(row, width, height, 0.22)
            destination = (
                args.output / split / "building" / f"{image_path.stem}_b{index:02d}_context.jpg"
            )
            counts[split]["building"] += int(save_crop(image, context_box, destination))

            exact_box = yolo_box(row, width, height, 0.02)
            for kind, partial_box in partial_building_boxes(exact_box):
                destination = (
                    args.output
                    / split
                    / "not_building"
                    / f"{image_path.stem}_b{index:02d}_{kind}.jpg"
                )
                counts[split]["not_building"] += int(save_crop(image, partial_box, destination))

        for index, row in enumerate(row for row in rows if row.split()[0] != str(BUILDING_CLASS)):
            box = yolo_box(row, width, height, 0.06)
            negative_candidates[split].append((image_path, box, f"label{index:02d}"))

        # Add background proposals with building-like aspect ratios.  Reject
        # any crop that intersects a labeled building, so positives never leak
        # into the negative class.
        for index in range(3):
            bw = rng.randint(round(width * 0.14), round(width * 0.38))
            bh = rng.randint(round(height * 0.09), round(height * 0.26))
            x1 = rng.randint(0, width - bw)
            y1 = rng.randint(0, height - bh)
            box = (x1, y1, x1 + bw, y1 + bh)
            if all(iou(box, positive) < 0.02 for positive in building_boxes):
                negative_candidates[split].append((image_path, box, f"random{index:02d}"))

    # Add one background/non-building crop per positive crop. Partial-building
    # hard negatives above already provide the remaining structural contrast.
    for split in ("train", "val"):
        rng.shuffle(negative_candidates[split])
        limit = min(len(negative_candidates[split]), counts[split]["building"])
        for index, (image_path, box, source) in enumerate(negative_candidates[split][:limit]):
            image = Image.open(image_path).convert("RGB")
            destination = (
                args.output
                / split
                / "not_building"
                / f"{image_path.stem}_{source}_n{index:04d}.jpg"
            )
            counts[split]["not_building"] += int(save_crop(image, box, destination))

    summary = {"source": str(args.source.resolve()), "seed": args.seed, "counts": counts}
    temporary = args.output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
