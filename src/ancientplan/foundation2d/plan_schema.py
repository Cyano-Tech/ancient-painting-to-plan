"""Strict schemas for a relative, non-metric whole-scene ground plan."""

from copy import deepcopy
import math
from .grounding import convex_quad


def normalize_plan_envelope(stage, data, context):
    """Normalize only unambiguous metadata, never geometry or model decisions.

    A missing stage can be supplied from the request. The observed word-order
    variant ``partial_visible`` means the existing ``visible_partial`` enum.
    Explicitly different stages and all substantive schema errors stay rejected.
    Return a copy plus an audit trail only if the entire normalized result passes.
    """
    if not isinstance(data, dict):
        return data, []
    normalized = deepcopy(data)
    changes = []
    if "stage" not in normalized:
        normalized["stage"] = stage
        changes.append({"path": "stage", "before": None, "after": stage})
    if stage == "plan_buildings" and isinstance(normalized.get("objects"), list):
        for i, obj in enumerate(normalized["objects"]):
            if isinstance(obj, dict) and obj.get("anchor_status") == "partial_visible":
                obj["anchor_status"] = "visible_partial"
                changes.append(
                    {
                        "path": f"objects[{i}].anchor_status",
                        "before": "partial_visible",
                        "after": "visible_partial",
                    }
                )
    if not changes or validate_plan_stage(stage, normalized, context):
        return data, []
    return normalized, changes


def point(p):
    return (
        isinstance(p, list)
        and len(p) == 2
        and all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and 0 <= v <= 1000
            for v in p
        )
    )


def polygon(p, n=3):
    return isinstance(p, list) and len(p) >= n and all(point(x) for x in p)


def bbox(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and point(b[:2])
        and point(b[2:])
        and b[0] < b[2]
        and b[1] < b[3]
    )


def ratio(r):
    return (
        isinstance(r, list)
        and len(r) == 2
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in r
        )
        and 0 < r[0] <= r[1] <= 30
    )


def validate_pending(plan):
    issues = []
    seen = {b["id"] for b in plan["layout"]["buildings"]}
    for c in plan.get("pending_candidates", []):
        if not isinstance(c.get("id"), str) or c["id"] in seen:
            issues.append("duplicate/invalid pending ID")
        seen.add(c.get("id"))
        if not bbox(c.get("source_bbox")) or not point(c.get("plan_marker")):
            issues.append("invalid pending geometry")
        if c.get("candidate_only") is not True or c.get("marker_is_foundation") is not False:
            issues.append("pending candidate must not claim a foundation")
        if any(k in c for k in ("plan_size", "source_ids", "zone_id")):
            issues.append("pending candidate must not own physical area or source assignments")
        for key in ("label", "evidence", "uncertainty", "source_run"):
            if not isinstance(c.get(key), str) or not c[key]:
                issues.append("missing pending " + key)
    return issues


def validate_building(b, zones=None):
    issues = []
    if not isinstance(b.get("id"), str) or not b["id"]:
        issues.append("missing building ID")
    if b.get("category") not in {"house", "other"} or b.get("confidence") not in {
        "high",
        "medium",
        "low",
    }:
        issues.append(f"{b.get('id')}: category/confidence")
    if (
        not bbox(b.get("source_bbox"))
        or not point(b.get("source_anchor"))
        or not convex_quad(b.get("source_footprint"))
    ):
        issues.append(f"{b.get('id')}: invalid source geometry")
    if not point(b.get("plan_center")) or not ratio(b.get("ratio_range")):
        issues.append(f"{b.get('id')}: invalid plan/ratio")
    size = b.get("plan_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 < v < 500 for v in size)
    ):
        issues.append(f"{b.get('id')}: invalid plan size")
    elif (
        ratio(b.get("ratio_range"))
        and not b["ratio_range"][0] - 0.01 <= size[0] / size[1] <= b["ratio_range"][1] + 0.01
    ):
        issues.append(f"{b.get('id')}: chosen ratio outside declared interval")
    if not isinstance(b.get("front_clock"), (int, float)) or not 0 <= b["front_clock"] <= 12:
        issues.append(f"{b.get('id')}: invalid facing")
    if zones is not None and b.get("zone_id") not in zones:
        issues.append(f"{b.get('id')}: unknown support zone")
    if not isinstance(b.get("source_ids"), list) or not all(
        isinstance(i, str) for i in b["source_ids"]
    ):
        issues.append(f"{b.get('id')}: invalid source IDs")
    for key in ("evidence", "assumption", "label", "subtype"):
        if not isinstance(b.get(key), str) or not b[key]:
            issues.append(f"{b.get('id')}: missing {key}")
    if b.get("observation_lock"):
        fields = b["observation_lock"].get("fields", {})
        if any(
            b.get(k) != fields.get(k) for k in ("source_bbox", "source_anchor", "source_footprint")
        ):
            issues.append(f"{b.get('id')}: changed locked source observations")
    return issues


def validate_terrain(data):
    issues = []
    if not polygon(data.get("source_boundary")) or not polygon(data.get("plan_boundary")):
        issues.append("invalid scene boundary")
    ids = set()
    for key in ("terrain", "zones", "trees", "routes"):
        if not isinstance(data.get(key), list):
            issues.append(f"{key} is not a list")
            continue
        for obj in data[key]:
            oid = obj.get("id")
            if not isinstance(oid, str) or oid in ids:
                issues.append(f"{key}: missing/duplicate ID")
            ids.add(oid)
            if key in {"terrain", "zones"}:
                if not polygon(obj.get("source_polygon")) or not polygon(obj.get("plan_polygon")):
                    issues.append(f"{oid}: invalid polygon")
                if obj.get("level") not in {"low", "middle", "high"}:
                    issues.append(f"{oid}: level")
                if key == "terrain" and obj.get("category") not in {
                    "mountain",
                    "water",
                    "flat",
                    "other",
                }:
                    issues.append(f"{oid}: category")
                if key == "zones":
                    axes = obj.get("ground_axes_image")
                    if (
                        not isinstance(axes, list)
                        or len(axes) != 2
                        or any(
                            not isinstance(a, list)
                            or len(a) != 2
                            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in a)
                            or math.hypot(*a) < 1e-6
                            for a in axes
                        )
                    ):
                        issues.append(f"{oid}: axes")
            else:
                a, b = (
                    ("source_roots", "plan_roots")
                    if key == "trees"
                    else ("source_points", "plan_points")
                )
                if not polygon(obj.get(a), 1 if key == "trees" else 2) or not polygon(
                    obj.get(b), 1 if key == "trees" else 2
                ):
                    issues.append(f"{oid}: invalid points")
    return issues


def validate_plan_stage(stage, data, context):
    try:
        issues = []
        if data.get("stage") != stage:
            issues.append("wrong stage")
        if stage == "plan_terrain":
            return issues + validate_terrain(data)
        if stage == "plan_buildings":
            ids = set()
            for b in data["objects"]:
                if not isinstance(b.get("id"), str) or b["id"] in ids:
                    issues.append("duplicate/missing candidate ID")
                ids.add(b.get("id"))
                if (
                    not bbox(b.get("bbox"))
                    or not point(b.get("anchor"))
                    or not convex_quad(b.get("footprint_image"))
                    or not ratio(b.get("width_depth_ratio"))
                ):
                    issues.append(f"{b.get('id')}: invalid candidate geometry")
                if b.get("category") not in {"house", "other"} or b.get("anchor_status") not in {
                    "visible_partial",
                    "inferred",
                    "outside_crop",
                }:
                    issues.append(f"{b.get('id')}: semantic fields")
            if not isinstance(data.get("excluded"), list) or not isinstance(
                data.get("coverage_notes"), str
            ):
                issues.append("missing coverage/exclusions")
        elif stage == "plan_layout":
            candidates = {b["id"] for b in context["candidates"]}
            zones = {z["id"] for z in context["terrain"]["zones"]}
            # Access/adjacency can legitimately refer to a route or tree group,
            # not only a land polygon. All targets must still exist in context.
            terrain_ids = {
                z["id"]
                for key in ("terrain", "trees", "routes")
                for z in context["terrain"].get(key, [])
            }
            ids = set()
            seen = []
            for b in data["buildings"]:
                issues += validate_building(b, zones)
                if b["id"] in ids:
                    issues.append("duplicate building ID")
                ids.add(b["id"])
                seen += b["source_ids"]
                if not b["source_ids"] and "newly_observed" not in b["evidence"]:
                    issues.append(f"{b['id']}: undocumented addition")
            seen += [r["source_id"] for r in data["rejected"]]
            if set(seen) != candidates or len(seen) != len(set(seen)):
                issues.append("Every source candidate must be accounted for exactly once")
            for r in data["relations"]:
                if r["from"] not in ids or r["to"] not in ids | zones | terrain_ids:
                    issues.append("unknown relation endpoint")
        elif stage == "plan_critique":
            if data.get("verdict") not in {"reasonable_approximation", "revise"}:
                issues.append("invalid verdict")
            for key in (
                "checks",
                "issues",
                "replace_buildings",
                "remove_buildings",
                "add_buildings",
                "replace_terrain",
                "replace_zones",
                "replace_trees",
                "replace_routes",
                "remaining_uncertainties",
            ):
                if not isinstance(data.get(key), list):
                    issues.append(f"missing {key}")
            original = {b["id"] for b in context["layout"]["buildings"]}
            newids = set()
            remove = {b["id"] for b in data["remove_buildings"]}
            for b in data["replace_buildings"]:
                issues += validate_building(b)
                if b["id"] not in original or b["id"] in newids or b["id"] in remove:
                    issues.append("invalid replacement ID")
                newids.add(b["id"])
            for b in data["add_buildings"]:
                issues += validate_building(b)
                if (
                    b["id"] in original | newids
                    or b["source_ids"]
                    or "newly_observed" not in b["evidence"]
                ):
                    issues.append("invalid added building")
                newids.add(b["id"])
            for b in data["remove_buildings"]:
                if (
                    b["id"] not in original
                    or b.get("merge_into") in remove
                    or (
                        b.get("merge_into") is not None and b["merge_into"] not in original | newids
                    )
                ):
                    issues.append("invalid removal/merge")
        elif stage == "plan_details":
            expected = {b["id"] for b in context["candidates"]}
            seen = []
            retained = {
                b["id"] for b in data["decisions"] if b.get("status") in {"keep", "uncertain"}
            }
            for b in data["decisions"] + data["missing"]:
                if (
                    not bbox(b.get("bbox"))
                    or not point(b.get("anchor"))
                    or not convex_quad(b.get("footprint_image"))
                    or not ratio(b.get("ratio_range"))
                ):
                    issues.append(f"{b.get('id')}: invalid detail geometry")
            for b in data["decisions"]:
                seen.append(b["id"])
                if b.get("status") not in {"keep", "uncertain", "merge", "reject"}:
                    issues.append("invalid detail verdict")
                if b.get("status") == "merge" and b.get("merge_into") not in retained:
                    issues.append("invalid merge target")
                if context.get("identity_audit"):
                    for key in ("foundation_trace", "bbox_reason"):
                        if not isinstance(b.get(key), str) or not b[key].strip():
                            issues.append(f"{b['id']}: missing {key}")
                    allowed = {
                        "keep": {"independent"},
                        "uncertain": {"independent", "unresolved"},
                        "merge": {"shared"},
                        "reject": {"nonbuilding"},
                    }
                    if b.get("independence") not in allowed.get(b.get("status"), set()):
                        issues.append(f"{b['id']}: inconsistent foundation identity")
                    if b.get("ground_contact") not in {"visible_partial", "inferred"}:
                        issues.append(f"{b['id']}: missing ground contact status")
                    if "appearance_kind" in b:
                        if b["appearance_kind"] not in {"physical", "reflection", "unclear"}:
                            issues.append(f"{b['id']}: invalid appearance kind")
                        if b["appearance_kind"] == "reflection" and (
                            b["status"] != "reject" or b.get("merge_into") is not None
                        ):
                            issues.append(
                                f"{b['id']}: reflection cannot own or merge a physical foundation"
                            )
                        if b["status"] == "keep" and b["appearance_kind"] != "physical":
                            issues.append(f"{b['id']}: unconfirmed physical building")
            if set(seen) != expected or len(seen) != len(expected):
                issues.append("detail input accounting failed")
        elif stage == "plan_accounting":
            expected = set(context["expected_source_ids"])
            ids = {b["id"] for b in context["layout"]["buildings"]}
            seen = []
            for a in data["assignments"]:
                seen.append(a["source_id"])
                if a["target"] is not None and a["target"] not in ids:
                    issues.append("unknown assignment target")
            if set(seen) != expected or len(seen) != len(expected):
                issues.append("assignment accounting failed")
        elif stage == "plan_final_check":
            if data.get("verdict") not in {"ready_for_confirmation", "needs_revision"}:
                issues.append("invalid final verdict")
            for key in ("blocking_issues", "coverage", "uncertainties"):
                if not isinstance(data.get(key), list):
                    issues.append("missing " + key)
            if data.get("verdict") == "ready_for_confirmation" and data.get("blocking_issues"):
                issues.append("ready verdict with blocking issues")
            if "geometric_facts" in context:
                for key in ("active_footprint_regions", "pending_candidates"):
                    if data.get("verified_counts", {}).get(key) != context["geometric_facts"][key]:
                        issues.append("final count differs from current snapshot: " + key)
        else:
            issues.append("unknown plan stage")
        return issues
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return ["Malformed plan data: " + type(exc).__name__]
