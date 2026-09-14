"""Independently re-audit a missing candidate before promoting it to a plan.

The staged plan is explicitly non-publishable. Promotion uses accepted crop
evidence; its newly inferred local support patch and relative placement are
layout hypotheses, not measured source geometry.
"""

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .complete_plan import read, save, stamp
from .apply_identity_evidence import load_evidence, revise
from .render_complete_plan import rect_corners
from .qwen_cloud import SafeError


def stage(plan_path, manifest_path, run, missing_id, new_id):
    p = read(plan_path)
    m = read(manifest_path)
    if m["source"] != p["source"]:
        raise SafeError("Unrelated missing observation")
    if new_id in {b["id"] for b in p["layout"]["buildings"]}:
        raise SafeError("ID already exists")
    observations = [
        d
        for e in m["evidence"]
        if Path(e["run"]).resolve() == Path(run).resolve()
        for d in e["missing"]
        if d["id"] == missing_id
    ]
    if len(observations) != 1:
        raise SafeError("Ambiguous missing observation")
    e = observations[0]
    b = {
        "id": new_id,
        "source_ids": [],
        "category": e["category"],
        "subtype": e["subtype"],
        "label": "待复核新候选",
        "source_bbox": e["bbox"],
        "source_anchor": e["anchor"],
        "source_footprint": e["footprint_image"],
        "plan_center": e["anchor"],
        "plan_size": [10, 10],
        "front_clock": e["front_clock"],
        "ratio_range": [1, 1],
        "zone_id": p["terrain"]["zones"][0]["id"],
        "confidence": "low",
        "evidence": "newly_observed provisional candidate; NOT AN ACCEPTED BUILDING",
        "assumption": "audit only",
    }
    p["layout"]["buildings"].append(b)
    p["staged_missing"] = {
        "id": new_id,
        "parent_plan": str(Path(plan_path).resolve()),
        "parent_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "evidence_manifest": str(Path(manifest_path).resolve()),
        "source_run": str(Path(run).resolve()),
        "missing_id": missing_id,
    }
    p["review_only"] = True
    folder = stamp("missing_candidate_audit_only")
    save(folder / "plan.json", p)
    print(folder / "plan.json", flush=True)
    return folder


def promote(staged_path, manifest_path):
    staged = read(staged_path)
    record = staged["staged_missing"]
    parent = Path(record["parent_plan"])
    old = read(parent)
    if hashlib.sha256(parent.read_bytes()).hexdigest() != record["parent_sha256"]:
        raise SafeError("Parent snapshot changed")
    evidence, missing = load_evidence(staged_path, manifest_path)
    oid = record["id"]
    if (
        set(evidence) != {oid}
        or evidence[oid]["status"] != "keep"
        or evidence[oid].get("appearance_kind") != "physical"
    ):
        raise SafeError("Missing candidate not independently confirmed as physical")
    updated, _ = revise(staged, evidence)
    b = deepcopy(next(b for b in updated["layout"]["buildings"] if b["id"] == oid))
    e = evidence[oid]
    foot = b["source_footprint"]
    sx, sy = b["source_anchor"]
    # Use local existing source->plan displacement, not object height or bbox
    # aspect. Width is the inferred footprint's larger image-plane span; depth
    # comes from the independent aspect interval, not perspective foreshortening.
    nearest = min(old["layout"]["buildings"], key=lambda x: math.dist(x["source_anchor"], [sx, sy]))
    shift = [nearest["plan_center"][i] - nearest["source_anchor"][i] for i in (0, 1)]
    if math.hypot(*shift) > 50:
        shift = [0, 0]
    b["plan_center"] = [round(sx + shift[0], 3), round(sy + shift[1], 3)]
    width = max(max(p[i] for p in foot) - min(p[i] for p in foot) for i in (0, 1))
    b["plan_size"] = [round(width, 3), round(width / (sum(e["ratio_range"]) / 2), 3)]
    b["ratio_range"] = e["ratio_range"]
    b["evidence"] = "newly_observed: " + e["evidence"]
    b["confidence"] = "low"
    b["assumption"] = (
        e["assumption"] + " 新增局部支承片、相对平移和尺度为布局假设；不是测绘得到的台地边界。"
    )
    zid = "Z_" + oid
    b["zone_id"] = zid
    corners = rect_corners(b)

    def envelope(points, pad):
        x0, y0 = [max(0, min(p[i] for p in points) - pad) for i in (0, 1)]
        x1, y1 = [min(1000, max(p[i] for p in points) + pad) for i in (0, 1)]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    zone = {
        "id": zid,
        "label": "新增屋盖的局部支承假设",
        "source_polygon": envelope(foot, 4),
        "plan_polygon": envelope(corners, 10),
        "level": "middle",
        "ground_axes_image": [
            [foot[1][i] - foot[0][i] for i in (0, 1)],
            [foot[3][i] - foot[0][i] for i in (0, 1)],
        ],
        "basis": "Independent physical roof/wall evidence; footprint hidden. Support patch is an inferred envelope, not an observed terrace boundary.",
    }
    output = deepcopy(old)
    output["layout"]["buildings"].append(b)
    output["terrain"]["zones"].append(zone)
    output.setdefault("promoted_missing", []).append(
        {
            "id": oid,
            "staged_plan": str(Path(staged_path).resolve()),
            "audit_manifest": str(Path(manifest_path).resolve()),
            "source_run": e["source_run"],
            "relative_placement_reference": nearest["id"],
            "relative_shift": shift,
            "metric_size_verified": False,
        }
    )
    output["parent_plan"] = str(parent)
    folder = stamp("whole_plan_missing_promoted")
    save(folder / "plan.json", output)
    print(folder / "plan.json", flush=True)
    return folder


def record_pending(staged_path, manifest_path):
    staged = read(staged_path)
    record = staged["staged_missing"]
    parent = Path(record["parent_plan"])
    old = read(parent)
    if hashlib.sha256(parent.read_bytes()).hexdigest() != record["parent_sha256"]:
        raise SafeError("Parent snapshot changed")
    evidence, _ = load_evidence(staged_path, manifest_path)
    oid = record["id"]
    if set(evidence) != {oid} or evidence[oid]["status"] != "uncertain":
        raise SafeError("Requires one unresolved candidate")
    e = evidence[oid]
    output = deepcopy(old)
    if oid in {c["id"] for c in output.get("pending_candidates", [])}:
        raise SafeError("Pending ID already exists")
    # No foundation/area is claimed for an unconfirmed building. The source box
    # is visible, and the plan has only a question-marker at a relative location.
    center = [sum(p[i] for p in e["footprint_image"]) / 4 for i in (0, 1)]
    nearest = min(old["layout"]["buildings"], key=lambda x: math.dist(x["source_anchor"], center))
    shift = [nearest["plan_center"][i] - nearest["source_anchor"][i] for i in (0, 1)]
    if math.hypot(*shift) > 50:
        shift = [0, 0]
    output.setdefault("pending_candidates", []).append(
        {
            "id": oid,
            "label": e["label"],
            "source_bbox": e["bbox"],
            "plan_marker": [round(center[i] + shift[i], 3) for i in (0, 1)],
            "evidence": e["evidence"],
            "uncertainty": e["assumption"],
            "source_run": e["source_run"],
            "candidate_only": True,
            "marker_is_foundation": False,
            "independence": e["independence"],
        }
    )
    output["parent_plan"] = str(parent)
    folder = stamp("whole_plan_with_pending")
    save(folder / "plan.json", output)
    print(folder / "plan.json", flush=True)
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("stage")
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--manifest", required=True, type=Path)
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--missing-id", required=True)
    a.add_argument("--new-id", required=True)
    a = sub.add_parser("promote")
    a.add_argument("--staged-plan", required=True, type=Path)
    a.add_argument("--evidence", required=True, type=Path)
    a = sub.add_parser("record-pending")
    a.add_argument("--staged-plan", required=True, type=Path)
    a.add_argument("--evidence", required=True, type=Path)
    a = p.parse_args()
    if a.command == "stage":
        stage(a.plan, a.manifest, a.run, a.missing_id, a.new_id)
    elif a.command == "promote":
        promote(a.staged_plan, a.evidence)
    else:
        record_pending(a.staged_plan, a.evidence)
