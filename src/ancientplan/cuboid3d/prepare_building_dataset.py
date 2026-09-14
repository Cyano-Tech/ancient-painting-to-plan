#!/usr/bin/env python3
"""Create a reproducible building-only YOLO dataset from the CLP labels.

The source dataset uses class 1 for ``building``.  This script remaps it to
class 0 and deliberately keeps negative paintings: they are useful evidence
against mistaking mountains, fences, and tree masses for buildings.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def building_rows(label_path: Path) -> list[str]:
    rows: list[str] = []
    if not label_path.exists():
        return rows
    for row in label_path.read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if len(fields) == 5 and fields[0] == "1":
            rows.append("0 " + " ".join(fields[1:]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(".dataset_cache/datasets/edmundxu/chineselandscapeobjectdetection/versions/4"),
    )
    parser.add_argument("--output", type=Path, default=Path("training_data/building"))
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    images = sorted(args.source.joinpath("images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No source images found under {args.source}")

    shuffled = images.copy()
    random.Random(args.seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * args.validation_fraction)
    validation_names = {p.name for p in shuffled[:validation_count]}

    for split in ("train", "val"):
        args.output.joinpath("images", split).mkdir(parents=True, exist_ok=True)
        args.output.joinpath("labels", split).mkdir(parents=True, exist_ok=True)

    stats = {
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "building_source_class": 1,
        "building_target_class": 0,
        "train_images": 0,
        "validation_images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "building_instances": 0,
    }
    for image in images:
        split = "val" if image.name in validation_names else "train"
        destination_image = args.output / "images" / split / image.name
        if not destination_image.exists():
            shutil.copy2(image, destination_image)
        rows = building_rows(args.source / "labels" / f"{image.stem}.txt")
        atomic_write(
            args.output / "labels" / split / f"{image.stem}.txt",
            "\n".join(rows) + ("\n" if rows else ""),
        )
        stats["validation_images" if split == "val" else "train_images"] += 1
        stats["positive_images" if rows else "negative_images"] += 1
        stats["building_instances"] += len(rows)

    yaml = f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: building\n"
    atomic_write(args.output / "building.yaml", yaml)
    atomic_write(args.output / "summary.json", json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
