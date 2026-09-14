"""Map reviewed candidates to the original image, without assuming identity."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .advance_scene import load_prior


def to_source_bbox(box, crop):
    x0, y0, x1, y1 = crop
    a, b, c, d = box
    return [
        x0 + a * (x1 - x0) / 1000,
        y0 + b * (y1 - y0) / 1000,
        x0 + c * (x1 - x0) / 1000,
        y0 + d * (y1 - y0) / 1000,
    ]


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0


def summarize(run_dirs, output_dir):
    combined = []
    runs = []
    source = None
    for run_dir in run_dirs:
        meta, data, _ = load_prior(run_dir)
        if meta["stage"] != "review":
            raise ValueError("Only completed review runs are supported")
        im = meta["image"]
        if source and source["sha256"] != im["source_sha256"]:
            raise ValueError("Cannot combine unrelated paintings")
        source = {
            "path": im["source_path"],
            "sha256": im["source_sha256"],
            "size": im["source_size_after_exif"],
        }
        namespace = Path(run_dir).name
        for obj in data["objects"]:
            combined.append(
                {
                    "global_id": namespace + ":" + obj["id"],
                    "review_run": str(Path(run_dir).resolve()),
                    "id_in_review": obj["id"],
                    "source_ids_in_tile": obj["source_ids"],
                    "category": obj["category"],
                    "bbox_source_pixels": to_source_bbox(obj["bbox"], im["crop_in_source_pixels"]),
                    "review_status": obj["review_status"],
                    "evidence": obj["evidence"],
                }
            )
        runs.append(
            {
                "run": str(Path(run_dir).resolve()),
                "crop": im["crop_in_source_pixels"],
                "candidate_counts": dict(Counter(o["category"] for o in data["objects"])),
                "merged_source_groups": [
                    {"id": o["id"], "source_ids": o["source_ids"]}
                    for o in data["objects"]
                    if len(o["source_ids"]) > 1
                ],
                "rejected": data["rejected"],
                "uncertain_ids": [
                    o["id"] for o in data["objects"] if o["review_status"] == "uncertain"
                ],
            }
        )
    houses = [o for o in combined if o["category"] == "house"]
    pairs = []
    for i, a in enumerate(houses):
        for b in houses[i + 1 :]:
            overlap = iou(a["bbox_source_pixels"], b["bbox_source_pixels"])
            if overlap >= 0.12:
                pairs.append(
                    {
                        "ids": [a["global_id"], b["global_id"]],
                        "iou": round(overlap, 4),
                        "cross_tile": a["review_run"] != b["review_run"],
                        "decision": "unresolved",
                        "note": "Overlap is only a review trigger, not evidence of instance identity.",
                    }
                )
    report = {
        "source": source,
        "runs": runs,
        "objects": combined,
        "overlap_review_pairs": pairs,
        "global_instance_deduplication_done": False,
        "semantic_coverage_verified": False,
        "coordinate_mapping_only": True,
        "counts_are_not_unique_instances": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir / "review_summary.json"),
                "reviewed_tile_candidates": len(combined),
                "house_overlap_pairs_requiring_review": len(pairs),
                "unique_house_count": "not_established",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    summarize(a.runs, a.output)
