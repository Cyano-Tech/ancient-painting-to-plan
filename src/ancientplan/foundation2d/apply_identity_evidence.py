"""Apply image-only identity evidence separately from any plan layout edits.

Source geometry is reconstructed from the accepted raw response and its actual
crop, rather than trusting a hand-edited mapped manifest. No new/missing object
is silently added: those require a separate contextual audit before promotion.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from .complete_plan import accepted, map_point, read, save, stamp
from .apply_shape_evidence import constrain
from .plan_schema import validate_plan_stage, validate_terrain
from .qwen_cloud import SafeError
from .render_complete_plan import contains

OBSERVED_FIELDS = ("source_bbox", "source_anchor", "source_footprint")


def preserve_observation_locks(old, new):
    """A layout reviewer may neither move image landmarks nor drop their lock."""
    before = {b["id"]: b for b in old["layout"]["buildings"]}
    for b in new["layout"]["buildings"]:
        lock = before.get(b["id"], {}).get("observation_lock")
        if lock:
            if any(b[k] != lock["fields"][k] for k in OBSERVED_FIELDS):
                raise SafeError(
                    f"{b['id']}: layout tried to change locked source observations; run an independent image audit"
                )
            b["observation_lock"] = deepcopy(lock)
            for key in (
                "identity_audit",
                "reported_source_anchor",
                "anchor_derivation",
                "grouped",
                "unit_count_range",
            ):
                if key in before[b["id"]] and key not in b:
                    b[key] = deepcopy(before[b["id"]][key])


def revise(plan, evidence):
    output = deepcopy(plan)
    known = {b["id"]: b for b in output["layout"]["buildings"]}
    if not evidence.keys() <= known.keys():
        raise SafeError("Unknown audit identity")
    removed = {i: e for i, e in evidence.items() if e["status"] in {"merge", "reject"}}
    changes = []
    for oid, e in evidence.items():
        if oid in removed:
            continue
        b = known[oid]
        old = deepcopy(b)
        b.update(
            source_bbox=e["bbox"],
            source_anchor=e["anchor"],
            source_footprint=e["footprint_image"],
            category=e["category"],
            subtype=e["subtype"],
            front_clock=e["front_clock"],
            label=e.get("label") or e["subtype"] + "（结构复核）",
            confidence="low" if e["status"] == "uncertain" else "medium",
            evidence=e["evidence"],
            assumption=e["assumption"],
        )
        # anchor is an annotation handle at the inferred foundation, not another
        # free model landmark. Do not leave its label floating over the roof.
        if not contains(b["source_anchor"], b["source_footprint"]):
            b["reported_source_anchor"] = b["source_anchor"]
            b["source_anchor"] = [
                round(sum(p[i] for p in b["source_footprint"]) / 4, 3) for i in (0, 1)
            ]
            b["anchor_derivation"] = (
                "Mean of inferred convex foundation corners; raw model anchor was outside its own foundation."
            )
        b["identity_audit"] = {
            k: e[k]
            for k in (
                "status",
                "foundation_trace",
                "independence",
                "ground_contact",
                "bbox_reason",
                "source_run",
            )
        }
        b["identity_audit"].update(
            {k: e[k] for k in ("appearance_kind", "reflection_evidence") if k in e}
        )
        if e.get("appearance_kind") == "physical" and e["independence"] == "unresolved":
            b["grouped"] = True
            b["assumption"] += (
                " 当前以建筑占地区包络表示；是否单栋及附属前厦的拆分未定，不将一个编号当作准确栋数。"
            )
        b["observation_lock"] = {
            "source_sha256": plan["source"]["sha256"],
            "run": e["source_run"],
            "fields": {k: deepcopy(b[k]) for k in OBSERVED_FIELDS},
        }
        changes.append(
            {
                "id": oid,
                "action": e["status"],
                "before": old,
                "after_observation": deepcopy(b),
                "evidence_run": e["source_run"],
            }
        )
    for oid, e in removed.items():
        b = known.pop(oid)
        if e["status"] == "merge":
            target = e.get("merge_into")
            if target not in evidence or target in removed:
                raise SafeError("Unreviewed or removed merge target")
            known[target]["source_ids"] = sorted(set(known[target]["source_ids"] + b["source_ids"]))
        else:
            output["layout"]["rejected"] += [
                {"source_id": s, "reason": e["evidence"]} for s in b["source_ids"]
            ]
        changes.append(
            {
                "id": oid,
                "action": e["status"],
                "merge_into": e.get("merge_into"),
                "before": b,
                "evidence": e["evidence"],
                "foundation_trace": e["foundation_trace"],
                "evidence_run": e["source_run"],
            }
        )
    output["layout"]["buildings"] = list(known.values())
    redirects = {
        i: e.get("merge_into") if e["status"] == "merge" else None for i, e in removed.items()
    }
    relations = []
    for rel in output["layout"]["relations"]:
        rel = {
            **rel,
            "from": redirects.get(rel["from"], rel["from"]),
            "to": redirects.get(rel["to"], rel["to"]),
        }
        if rel["from"] and rel["to"] and rel["from"] != rel["to"]:
            relations.append(rel)
    output["layout"]["relations"] = relations
    output, shape_report = constrain(output, evidence)
    report = {
        "changes": changes,
        "shape_constraints": shape_report,
        "plan_centers_changed": False,
        "source_geometry_basis": "accepted image-only crop audit; never plan collisions",
        "missing_candidates_auto_added": False,
    }
    output["identity_audits"] = output.get("identity_audits", []) + [report]
    output.setdefault("revisions", []).append(
        {
            "verdict": "image_identity_audit",
            "run": sorted({e["source_run"] for e in evidence.values()}),
            "issues": [{"ids": [c["id"]], "action": c["action"]} for c in changes],
            "checks": [
                "Independent raw-image census followed by ID association; source geometry locked."
            ],
            "remaining_uncertainties": ["Occluded foundations and dimensions remain hypotheses."],
        }
    )
    return output, report


def load_evidence(plan_path, manifest_path):
    plan = read(plan_path)
    manifest = read(manifest_path)
    if (
        manifest.get("source") != plan["source"]
        or manifest.get("plan_sha256") != hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    ):
        raise SafeError("Audit belongs to a different source or plan snapshot")
    evidence = {}
    missing = []
    for job in manifest["jobs"]:
        if job.get("exit_code") != 0:
            raise SafeError("Incomplete audit")
        data = accepted(job["run"])
        meta = read(Path(job["run"]) / "request_meta.json")
        context = read(meta["context_source"])
        if not context.get("identity_audit"):
            raise SafeError("Requires an independent identity audit")
        if validate_plan_stage("plan_details", data, context):
            raise SafeError("Invalid independent audit schema")
        crop = job["crop"]
        # Check request provenance before using coordinates from the manifest.
        if (
            meta["image"].get("crop_in_source_pixels") != crop
            or meta["image"].get("source_sha256") != plan["source"]["sha256"]
        ):
            raise SafeError("Audit crop differs from request metadata")
        for d in data["decisions"]:
            if job.get("selected_ids") is not None and d["id"] not in job["selected_ids"]:
                continue
            if d["id"] in evidence:
                raise SafeError("Ambiguous duplicate audit decision")
            e = deepcopy(d)
            e["source_run"] = job["run"]
            e["bbox"] = map_point(d["bbox"][:2], crop, plan["source"]["size"]) + map_point(
                d["bbox"][2:], crop, plan["source"]["size"]
            )
            e["anchor"] = map_point(d["anchor"], crop, plan["source"]["size"])
            e["footprint_image"] = [
                map_point(p, crop, plan["source"]["size"]) for p in d["footprint_image"]
            ]
            evidence[e["id"]] = e
        missing.extend({"run": job["run"], "candidate": d} for d in data["missing"])
    return evidence, missing


def apply(plan_path, manifest_path):
    old = read(plan_path)
    evidence, missing = load_evidence(plan_path, manifest_path)
    output, report = revise(old, evidence)
    expected = {s for b in old["layout"]["buildings"] for s in b["source_ids"]} | {
        r["source_id"] for r in old["layout"]["rejected"]
    }
    issues = validate_terrain(output["terrain"]) + validate_plan_stage(
        "plan_layout",
        output["layout"],
        {"candidates": [{"id": s} for s in expected], "terrain": output["terrain"]},
    )
    if issues:
        raise SafeError("Invalid identity revision: " + str(issues))
    report["unpromoted_missing_candidates"] = missing
    output["parent_plan"] = str(Path(plan_path).resolve())
    folder = stamp("whole_plan_identity_revised")
    save(folder / "plan.json", output)
    save(folder / "identity_report.json", report)
    print(folder / "plan.json", flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    apply(args.plan, args.evidence)
