"""Reproduce curated examples on CPU with no credentials or model downloads."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

from ancientplan.cuboid3d import pipeline, validate_outputs
from ancientplan.foundation2d.grounding import rasterize
from ancientplan.foundation2d.plan_schema import (
    validate_pending,
    validate_plan_stage,
    validate_terrain,
)
from ancientplan.foundation2d.render_complete_plan import geometry_checks, render


REPOSITORY = Path(__file__).resolve().parents[1]


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def empty_output(output: Path) -> None:
    """Never overwrite or recursively clean user data."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def replay_cuboids(example: Path, output: Path) -> dict:
    empty_output(output)
    scenes = json.loads((example / "scene_priors.json").read_text())["scenes"]
    contract = json.loads((pipeline.ROOT / "contracts/output_contract.json").read_text())
    validator = Draft202012Validator(
        json.loads((pipeline.ROOT / "contracts/scene_schema.json").read_text())
    )
    result = {}
    folders = []
    for name, priors in scenes.items():
        folder, metrics = pipeline.process_scene(
            example / "inputs" / (name + ".png"),
            output,
            priors,
            example / "house_detection",
            example / "building_orientation",
        )
        errors = validate_outputs.validate_scene(folder, contract, validator)
        if errors or not metrics["accepted"]:
            raise ValueError(f"{name}: {errors or 'internal reconstruction checks failed'}")
        result[name] = {
            "instances": metrics["parsing_gate"]["whole_house_instances"],
            "contract_errors": errors,
            "accuracy_measured": False,
        }
        folders.append(folder)
    pipeline.make_batch_sheet(folders, output)
    save(output / "replay_report.json", result)
    return result


def replay_foundations(example: Path, output: Path, png: bool = True) -> dict:
    empty_output(output)
    original = json.loads((example / "plan.json").read_text())
    plan = deepcopy(original)
    image = (example / plan["source"]["path"]).resolve()
    if hashlib.sha256(image.read_bytes()).hexdigest() != plan["source"]["sha256"]:
        raise ValueError("Example source image checksum mismatch")
    source_ids = [sid for b in plan["layout"]["buildings"] for sid in b["source_ids"]]
    source_ids += [r["source_id"] for r in plan["layout"]["rejected"]]
    context = {
        "candidates": [{"id": sid} for sid in sorted(set(source_ids))],
        "terrain": plan["terrain"],
    }
    errors = (
        validate_terrain(plan["terrain"])
        + validate_plan_stage("plan_layout", plan["layout"], context)
        + validate_pending(plan)
    )
    geometry = geometry_checks(plan)
    if errors or any(g["severity"] == "major" for g in geometry):
        raise ValueError(f"Example validation failed: {errors or geometry}")
    shutil.copyfile(image, output / "source.jpg")
    plan["source"]["path"] = "source.jpg"
    save(output / "plan.json", plan)
    render(output / "plan.json", png=png)
    panes = []
    for i, name in enumerate(("source_overlay", "topview")):
        pane = ET.fromstring((output / (name + ".svg")).read_text())
        pane.set("x", str(i * 1400))
        pane.set("y", "0")
        panes.append(ET.tostring(pane, encoding="unicode"))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="2800" height="1510">'
        + "".join(panes)
        + "</svg>"
    )
    (output / "overview.svg").write_text(svg, encoding="utf-8")
    if png:
        rasterize(svg, output / "overview.png")
    result = {
        "mode": "offline_snapshot_replay",
        "gpu_used": False,
        "cloud_calls": 0,
        "footprint_regions": len(plan["layout"]["buildings"]),
        "categories": dict(Counter(b["category"] for b in plan["layout"]["buildings"])),
        "grouped_ids": [b["id"] for b in plan["layout"]["buildings"] if b.get("grouped")],
        "pending_ids": [b["id"] for b in plan.get("pending_candidates", [])],
        "source_candidates_accounted_for": len(source_ids),
        "schema_errors": errors,
        "geometry_issues": geometry,
        "semantic_accuracy_measured": False,
        "historical_review_rerun": False,
        "original_export_sha256": hashlib.sha256((example / "plan.json").read_bytes()).hexdigest(),
    }
    save(output / "replay_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("cuboid3d", "foundation2d", "all"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixtures",
        "--examples",
        dest="examples",
        type=Path,
        default=REPOSITORY / "tests" / "fixtures",
        help="offline replay fixtures (not the PNG-only showcase)",
    )
    parser.add_argument(
        "--svg-only", action="store_true", help="skip optional CairoSVG PNG export for foundation2d"
    )
    args = parser.parse_args()
    empty_output(args.output)
    results = {}
    if args.method in ("cuboid3d", "all"):
        results["cuboid3d"] = replay_cuboids(args.examples / "cuboid3d", args.output / "cuboid3d")
    if args.method in ("foundation2d", "all"):
        results["foundation2d"] = replay_foundations(
            args.examples / "foundation2d", args.output / "foundation2d", not args.svg_only
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
