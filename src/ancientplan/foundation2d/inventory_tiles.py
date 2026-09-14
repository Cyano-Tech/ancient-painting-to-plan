"""Four overlapping image crops, two concurrent Qwen cloud requests at most.

Each tile returns independent candidates. The manifest explicitly does not
claim globally deduplicated instances or verified semantic coverage.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image, ImageOps

from .qwen_cloud import RUN_ROOT, SafeError, run


def tile_boxes(width, height):
    midx, midy = width / 2, height / 2
    padx, pady = width * 0.06, height * 0.06
    return [
        ("nw", [0, 0, round(midx + padx), round(midy + pady)]),
        ("ne", [round(midx - padx), 0, width, round(midy + pady)]),
        ("sw", [0, round(midy - pady), round(midx + padx), height]),
        ("se", [round(midx - padx), round(midy - pady), width, height]),
    ]


def execute_tile(image_path, tile):
    label, crop = tile
    args = argparse.Namespace(
        stage="inventory", image=str(image_path), crop=crop, base_url=None, run_label=label
    )
    try:
        code = run(args)
        record = {
            "tile": label,
            "crop_source_pixels": crop,
            "exit_code": code,
            "output_dir": args.output_dir,
        }
        result = json.loads((Path(args.output_dir) / "result.json").read_text())
        record.update(
            {
                "model_returned": result["model_returned"],
                "usage": result["usage"],
                "finish_reason": result["finish_reason"],
                "validation_issues": result["validation_issues"],
            }
        )
        if code == 0:
            objects = result["parsed"]["objects"]
            record["raw_candidate_counts"] = dict(Counter(o["category"] for o in objects))
            record["candidate_count"] = len(objects)
        return record
    except (SafeError, OSError, ValueError) as exc:
        return {
            "tile": label,
            "crop_source_pixels": crop,
            "exit_code": 1,
            "output_dir": getattr(args, "output_dir", None),
            "error_type": type(exc).__name__,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    image_path = args.image.resolve()
    with Image.open(image_path) as im:
        width, height = ImageOps.exif_transpose(im).size
    boxes = tile_boxes(width, height)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch = RUN_ROOT / (stamp + "_tiles")
    batch.mkdir(parents=True, mode=0o700)
    manifest = {
        "source_path": str(image_path),
        "source_size": [width, height],
        "tile_grid": "2x2",
        "overlap_fraction_of_full_image": 0.12,
        "max_parallel_requests": 2,
        "tile_rectangles_cover_full_image": True,
        "semantic_coverage_verified": False,
        "global_instance_deduplication_done": False,
        "tiles": [],
    }
    print("Batch manifest:", batch / "manifest.json", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda tile: execute_tile(image_path, tile), boxes):
            manifest["tiles"].append(result)
            (batch / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print("Tile completed:", result["tile"], "exit", result["exit_code"], flush=True)
    manifest["all_tiles_returned_valid_json"] = all(r["exit_code"] == 0 for r in manifest["tiles"])
    (batch / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(batch / "manifest.json"),
                "all_tiles_returned_valid_json": manifest["all_tiles_returned_valid_json"],
                "raw_candidates": sum(r.get("candidate_count", 0) for r in manifest["tiles"]),
                "deduplicated": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["all_tiles_returned_valid_json"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
