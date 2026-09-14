"""Package a visually reviewed whole plan without overwriting earlier results."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .complete_plan import accepted, read, save
from .grounding import rasterize
from .plan_schema import validate_plan_stage, validate_terrain, validate_pending
from .render_complete_plan import geometry_checks, render


def publish(plan_path, review_run, output):
    plan_path = Path(plan_path).resolve()
    review_run = Path(review_run).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing publication")
    plan = read(plan_path)
    review = accepted(review_run)
    meta = read(review_run / "request_meta.json")
    context = read(meta["context_source"])
    if plan.get("review_only") or plan.get("staged_missing"):
        raise ValueError("Provisional missing-candidate audit cannot be published")
    if context.get("plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError("Review is not for this exact plan")
    expected_verdict = (
        "ready_for_confirmation"
        if meta["stage"] == "plan_final_check"
        else "reasonable_approximation"
    )
    if review.get("verdict") != expected_verdict:
        raise ValueError("Final visual review still asks for revision")
    if any(
        review.get(key)
        for key in (
            "replace_buildings",
            "remove_buildings",
            "add_buildings",
            "replace_terrain",
            "replace_zones",
            "replace_trees",
            "replace_routes",
            "remove_elements",
        )
    ):
        raise ValueError("Unapplied visual review changes remain")
    if review.get("replace_rejected") is not None or review.get("replace_relations") is not None:
        raise ValueError("Unapplied metadata edits remain")
    expected = {s for b in plan["layout"]["buildings"] for s in b["source_ids"]} | {
        r["source_id"] for r in plan["layout"]["rejected"]
    }
    issues = (
        validate_terrain(plan["terrain"])
        + validate_plan_stage(
            "plan_layout",
            plan["layout"],
            {"candidates": [{"id": s} for s in expected], "terrain": plan["terrain"]},
        )
        + validate_pending(plan)
    )
    if issues or plan.get("draft_validation_issues"):
        raise ValueError("Invalid plan schema")
    geometry = geometry_checks(plan)
    if any(g["severity"] == "major" for g in geometry):
        raise ValueError("Major geometry conflicts remain")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes(plan_path.read_bytes())
    render(output / "plan.json", png=True)
    (output / "final_visual_review.json").write_bytes((review_run / "result.json").read_bytes())
    panes = []
    for i, name in enumerate(("source_overlay", "topview")):
        root = ET.fromstring((output / (name + ".svg")).read_text())
        root.set("x", str(i * 1400))
        root.set("y", "0")
        panes.append(ET.tostring(root, encoding="unicode"))
    overview = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="2800" height="1510">'
        + "".join(panes)
        + "</svg>"
    )
    (output / "overview.svg").write_text(overview)
    rasterize(overview, output / "overview.png")
    manifest = {
        "status": "ready_for_user_confirmation",
        "plan_source": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "final_review_run": str(review_run),
        "source_image_sha256": plan["source"]["sha256"],
        "cloud_model": "qwen3.8-max",
        "local_gpu_used": False,
        "geometry_issues": geometry,
        "semantic_truth_verified": False,
        "metric_calibration": False,
        "pending_candidate_ids": [c["id"] for c in plan.get("pending_candidates", [])],
        "meaning": "Complete relative ground-layout hypothesis, reviewed for evident scene contradictions; final acceptance belongs to the user.",
        "files": [
            "index.html",
            "overview.png",
            "topview.png",
            "source_overlay.png",
            "topview.svg",
            "plan.json",
            "geometry_report.json",
            "final_visual_review.json",
        ],
    }
    save(output / "publication_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    publish(a.plan, a.review, a.output)
