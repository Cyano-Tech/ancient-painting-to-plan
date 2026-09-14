"""Keep independent visual shape estimates separate from layout packing.

An invalidly elongated layout cannot validate itself by widening its own ratio
range. Use the independently inspected crop's range, retain the layout width,
and choose the range midpoint only when the current aspect lies outside it.
This is a hypothesis selection rule, not recovery of metric building size.
"""

import argparse
from copy import deepcopy
from pathlib import Path

from .complete_plan import accepted, read, save, stamp
from .plan_schema import validate_plan_stage, validate_terrain


def constrain(plan, evidence):
    output = deepcopy(plan)
    changes = []
    for b in output["layout"]["buildings"]:
        e = evidence.get(b["id"])
        if e is None or e["status"] not in {"keep", "uncertain"} or b.get("grouped"):
            continue
        low, high = e["ratio_range"]
        old = b["plan_size"][:]
        before_range = b["ratio_range"][:]
        actual = old[0] / old[1]
        if not low <= actual <= high:
            b["plan_size"] = [old[0], round(old[0] / ((low + high) / 2), 4)]
        b["ratio_range"] = [low, high]
        if old != b["plan_size"] or before_range != b["ratio_range"]:
            b["superseded_layout_shape_text"] = {
                "evidence": b["evidence"],
                "assumption": b["assumption"],
            }
            b["evidence"] = e["evidence"]
            b["assumption"] = (
                e.get("assumption", b["assumption"])
                + " 平面宽度沿用总图的相对尺度；进深受独立局部比例区间约束，不为避碰压薄。"
            )
            changes.append(
                {
                    "id": b["id"],
                    "size_before": old,
                    "size_after": b["plan_size"],
                    "ratio_before": before_range,
                    "independent_ratio": [low, high],
                    "source_run": e["source_run"],
                    "basis": e["evidence"],
                    "rule": "Width retained. Out-of-range aspect replaced by independent interval midpoint; no geometry-derived widening of the evidence range.",
                }
            )
    report = {
        "changes": changes,
        "source_geometry_changed": False,
        "identity_or_facing_changed": False,
        "metric_dimensions_verified": False,
    }
    output["shape_evidence_selections"] = output.get("shape_evidence_selections", []) + [report]
    return output, report


def apply(plan_path, manifest_path):
    plan = read(plan_path)
    manifest = read(manifest_path)
    if plan["source"] != manifest["source"]:
        raise ValueError("Unrelated shape evidence")
    evidence = {}
    for job in manifest["jobs"]:
        data = accepted(job["run"])
        if data["stage"] != "plan_details":
            raise ValueError("Requires independent detailed image analysis")
        for e in data["decisions"]:
            if e["id"] in evidence:
                raise ValueError("Ambiguous repeated detail estimates")
            evidence[e["id"]] = {**e, "source_run": job["run"]}
    output, report = constrain(plan, evidence)
    expected = {s for b in output["layout"]["buildings"] for s in b["source_ids"]} | {
        r["source_id"] for r in output["layout"]["rejected"]
    }
    issues = validate_terrain(output["terrain"]) + validate_plan_stage(
        "plan_layout",
        output["layout"],
        {"candidates": [{"id": s} for s in expected], "terrain": output["terrain"]},
    )
    if issues:
        raise ValueError("Invalid constrained plan: " + str(issues))
    folder = stamp("whole_plan_shape_constrained")
    output["parent_plan"] = str(Path(plan_path).resolve())
    save(folder / "plan.json", output)
    save(folder / "shape_selection_report.json", report)
    print(folder / "plan.json", flush=True)
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    apply(a.plan, a.evidence)
