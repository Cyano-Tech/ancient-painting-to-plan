"""Documented schema aliases only. Never change cloud coordinates or invent objects."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .qwen_cloud import validate_inventory


def normalize(data):
    result = deepcopy(data)
    changes = []
    for obj in result.get("objects", []):
        if isinstance(obj, dict) and obj.get("category") == "tree_group":
            changes.append(
                {
                    "id": obj.get("id"),
                    "from_category": "tree_group",
                    "to_category": "tree",
                    "grouped": True,
                    "reason": "Unambiguous tree-group alias; coordinates and evidence unchanged.",
                }
            )
            obj["category"] = "tree"
            obj["grouped"] = True
    return result, changes


def semantic_warnings(data):
    warnings = []
    objects = data.get("objects", [])
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if {a.get("category"), b.get("category")} == {"tree", "mountain"} and a.get(
                "bbox"
            ) == b.get("bbox"):
                warnings.append(
                    {
                        "type": "coincident_tree_mountain_boxes",
                        "ids": [a["id"], b["id"]],
                        "reason": "Same image box labeled both tree and mountain; requires visual review, not automatic relabeling.",
                    }
                )
    return warnings


def process(run_dir):
    raw = json.loads((run_dir / "result.json").read_text())
    if (
        raw.get("finish_reason") != "stop"
        or not raw.get("model_returned", "").startswith("qwen3.8-")
        or not isinstance(raw.get("parsed"), dict)
    ):
        raise ValueError("Cannot normalize a truncated, missing, or wrong-model response.")
    data, changes = normalize(raw["parsed"])
    issues = validate_inventory(data)
    report = {
        "source_result": "result.json",
        "model_returned": raw["model_returned"],
        "parsed": data,
        "normalization_changes": changes,
        "validation_issues": issues,
        "semantic_warnings": semantic_warnings(data),
        "coordinates_changed": False,
        "objects_added_or_removed": False,
        "vision_accuracy_verified": False,
        "raw_candidate_counts": dict(Counter(o["category"] for o in data.get("objects", []))),
    }
    path = run_dir / "inventory_normalized.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "normalizations": len(changes),
                "validation_issues": issues,
                "semantic_warnings": report["semantic_warnings"],
                "raw_candidate_counts": report["raw_candidate_counts"],
            },
            ensure_ascii=False,
        )
    )
    return not issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    success = True
    for run_dir in args.run_dirs:
        success = process(run_dir) and success
    raise SystemExit(0 if success else 2)
