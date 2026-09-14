"""Validate provenance and geometry without silently repairing model output."""

import math

from .qwen_cloud import validate_inventory


def point(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in value)
    )


def path(value, minimum=2):
    return isinstance(value, list) and len(value) >= minimum and all(point(p) for p in value)


def context_objects(context):
    if not isinstance(context, dict) or not isinstance(context.get("objects"), list):
        raise ValueError("Previous-stage objects are required.")
    return {o["id"]: o for o in context["objects"]}


def validate_review(data, context):
    issues = validate_inventory(data)
    if not isinstance(data.get("objects"), list):
        return issues
    sources = set(context_objects(context))
    accounted = set()
    retained = set()
    for obj in data["objects"]:
        if not isinstance(obj, dict):
            continue
        refs = obj.get("source_ids")
        if not isinstance(refs, list) or any(
            not isinstance(x, str) or x not in sources for x in refs
        ):
            issues.append(f"{obj.get('id')}: invalid source_ids")
        else:
            accounted.update(refs)
            retained.update(refs)
        if obj.get("review_status") not in {"supported", "uncertain"}:
            issues.append(f"{obj.get('id')}: invalid review_status")
    rejected = data.get("rejected")
    if not isinstance(rejected, list):
        issues.append("rejected must be a list")
        rejected = []
    reject_ids = set()
    for r in rejected:
        rid = r.get("source_id") if isinstance(r, dict) else None
        if not isinstance(rid, str) or rid not in sources or rid in reject_ids or rid in retained:
            issues.append("rejected has unknown/duplicate/retained source_id")
        else:
            reject_ids.add(rid)
            accounted.add(rid)
    if sources - accounted:
        issues.append("Unaccounted source IDs: " + ", ".join(sorted(sources - accounted)))
    return issues


def validate_structure(data, context):
    prior = context_objects(context)
    objects = data.get("objects")
    if not isinstance(objects, list):
        return ["objects must be a list"]
    issues, seen = [], set()
    planes = data.get("support_planes", [])
    if not isinstance(planes, list) or any(
        not isinstance(p, dict) or not isinstance(p.get("id"), str) for p in planes
    ):
        issues.append("Invalid support_planes")
        planes = []
    plane_ids = {p["id"] for p in planes}
    if len(plane_ids) != len(planes):
        issues.append("Duplicate support plane IDs")
    for obj in objects:
        if not isinstance(obj, dict):
            issues.append("Non-object structure entry")
            continue
        oid = obj.get("id")
        if not isinstance(oid, str) or oid not in prior or oid in seen:
            issues.append("Unknown or duplicate structure ID")
            continue
        seen.add(oid)
        if obj.get("category") != prior[oid]["category"]:
            issues.append(f"{oid}: category changed outside review stage")
        if obj.get("status") not in {"proposed", "deferred"}:
            issues.append(f"{oid}: invalid structure status")
        for name in ("visible_paths", "hidden_paths"):
            lines = obj.get(name)
            if not isinstance(lines, list) or any(not path(line) for line in lines):
                issues.append(f"{oid}: invalid {name}")
        for name in ("reference_ids", "occluded_by"):
            refs = obj.get(name)
            if not isinstance(refs, list) or any(
                not isinstance(x, str) or x not in prior or x == oid for x in refs
            ):
                issues.append(f"{oid}: invalid {name}")
        if obj.get("support_plane") is not None and obj.get("support_plane") not in plane_ids:
            issues.append(f"{oid}: unknown support plane")
        fp = obj.get("footprint")
        if not isinstance(fp, dict):
            issues.append(f"{oid}: missing footprint record")
            continue
        kind, points, states = fp.get("kind"), fp.get("points"), fp.get("edge_status")
        if kind not in {"point", "polyline", "polygon", "unknown"}:
            issues.append(f"{oid}: invalid footprint kind")
            continue
        if not isinstance(points, list) or not all(point(p) for p in points):
            issues.append(f"{oid}: invalid footprint points")
            continue
        n = len(points)
        expected = {"point": 1, "polyline": max(0, n - 1), "polygon": n, "unknown": 0}[kind]
        if (
            (kind == "point" and n != 1)
            or (kind == "polyline" and n < 2)
            or (kind == "polygon" and n < 3)
            or (kind == "unknown" and n)
        ):
            issues.append(f"{oid}: invalid footprint point count")
        if kind == "polygon" and n >= 3:
            area2 = sum(
                points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                for i in range(n)
            )
            if points[0] == points[-1] or abs(area2) < 1e-6:
                issues.append(f"{oid}: degenerate or duplicated closing point")
        if (
            not isinstance(states, list)
            or len(states) != expected
            or any(s not in {"visible", "inferred", "unknown"} for s in states)
        ):
            issues.append(f"{oid}: invalid edge provenance")
        if obj.get("status") == "deferred" and (kind != "unknown" or obj.get("hidden_paths")):
            issues.append(f"{oid}: deferred object must not have inferred geometry")
    if set(prior) != seen:
        issues.append("Structure did not account for every input object")
    return issues


def validate_audit(data, context):
    prior = context_objects(context)
    objects = data.get("objects")
    if not isinstance(objects, list):
        return ["objects must be a list"]
    seen, issues = set(), []
    for obj in objects:
        oid = obj.get("id") if isinstance(obj, dict) else None
        if not isinstance(oid, str) or oid not in prior or oid in seen:
            issues.append("Unknown/duplicate audit ID")
            continue
        seen.add(oid)
        if (
            obj.get("verdict") not in {"plausible", "needs_review", "reject"}
            or type(obj.get("safe_for_ground_projection")) is not bool
        ):
            issues.append(f"{oid}: invalid audit verdict")
        if obj.get("safe_for_ground_projection") and (
            obj.get("verdict") != "plausible"
            or prior[oid].get("status") == "deferred"
            or prior[oid].get("footprint", {}).get("kind") == "unknown"
        ):
            issues.append(f"{oid}: unsafe projection acceptance")
    if set(prior) != seen:
        issues.append("Audit did not account for every input object")
    return issues


def validate_stage(stage, data, context):
    try:
        return {
            "review": validate_review,
            "structure": validate_structure,
            "audit": validate_audit,
        }[stage](data, context)
    except (KeyError, TypeError, ValueError) as exc:
        return ["Malformed stage data: " + type(exc).__name__]
