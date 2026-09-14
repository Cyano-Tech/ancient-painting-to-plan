"""Decode JSON-string path arrays only; never repair coordinates or conclusions."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .scene_schema import path, validate_stage


def normalize(data):
    result = deepcopy(data)
    changes = []
    for obj in result.get("objects", []):
        for name in ("visible_paths", "hidden_paths"):
            lines = obj.get(name, [])
            if not isinstance(lines, list):
                continue
            for i, line in enumerate(lines):
                if not isinstance(line, str):
                    continue
                try:
                    decoded = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if path(decoded):
                    lines[i] = decoded
                    changes.append(
                        {
                            "id": obj["id"],
                            "field": name,
                            "index": i,
                            "from": "JSON string",
                            "to": "coordinate array",
                            "reason": "Lossless decoding of an explicitly JSON-encoded path; all numeric values preserved.",
                        }
                    )
    return result, changes


def process(run_dir):
    meta = json.loads((run_dir / "request_meta.json").read_text())
    raw = json.loads((run_dir / "result.json").read_text())
    if (
        meta["stage"] != "structure"
        or raw["finish_reason"] != "stop"
        or not raw["model_returned"].startswith("qwen3.8-")
        or not isinstance(raw["parsed"], dict)
    ):
        raise ValueError("Cannot normalize an incomplete or wrong-stage response")
    context = json.loads(Path(meta["context_source"]).read_text())["parsed"]
    data, changes = normalize(raw["parsed"])
    issues = validate_stage("structure", data, context)
    report = {
        "source_result": "result.json",
        "parsed": data,
        "normalization_changes": changes,
        "validation_issues": issues,
        "coordinates_changed": False,
        "objects_added_or_removed": False,
        "vision_accuracy_verified": False,
    }
    output = run_dir / "structure_normalized.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "format_changes": len(changes), "validation_issues": issues},
            ensure_ascii=False,
        )
    )
    return 0 if not issues else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    a = p.parse_args()
    raise SystemExit(process(a.run_dir))
