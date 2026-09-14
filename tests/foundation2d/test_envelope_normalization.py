"""Small format tolerances must not silently repair inference geometry."""

from copy import deepcopy

from ancientplan.foundation2d.plan_schema import normalize_plan_envelope, validate_plan_stage


def empty_crop():
    return {"objects": [], "excluded": [], "coverage_notes": "No visible buildings."}


def test_missing_stage_is_audited_without_mutating_raw():
    raw = empty_crop()
    before = deepcopy(raw)
    result, changes = normalize_plan_envelope("plan_buildings", raw, {})
    assert raw == before
    assert result == {**raw, "stage": "plan_buildings"}
    assert changes == [{"path": "stage", "before": None, "after": "plan_buildings"}]


def test_explicit_wrong_stage_not_repaired():
    raw = {**empty_crop(), "stage": "plan_terrain"}
    assert normalize_plan_envelope("plan_buildings", raw, {}) == (raw, [])


def test_invalid_payload_not_promoted():
    raw = {"objects": [], "excluded": []}
    assert normalize_plan_envelope("plan_buildings", raw, {}) == (raw, [])


def test_word_order_alias_never_changes_geometry():
    obj = {
        "id": "B1",
        "category": "house",
        "anchor_status": "partial_visible",
        "bbox": [10, 10, 50, 50],
        "anchor": [30, 45],
        "footprint_image": [[10, 30], [50, 30], [50, 50], [10, 50]],
        "width_depth_ratio": [1, 3],
    }
    raw = {**empty_crop(), "stage": "plan_buildings", "objects": [obj]}
    result, changes = normalize_plan_envelope("plan_buildings", raw, {})
    assert len(changes) == 1
    assert result["objects"][0] == {**obj, "anchor_status": "visible_partial"}
    assert raw["objects"][0]["anchor_status"] == "partial_visible"
    raw["objects"][0]["anchor"] = [-1, 45]
    assert normalize_plan_envelope("plan_buildings", raw, {}) == (raw, [])


def test_relation_can_target_existing_route_but_not_unknown_id():
    # Reuse an existing valid fixture so this tests relation endpoints only.
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "tests/fixtures/foundation2d/plan.json").read_text())
    layout = plan["layout"]
    source_ids = [sid for b in layout["buildings"] for sid in b["source_ids"]]
    source_ids += [r["source_id"] for r in layout["rejected"]]
    context = {"terrain": plan["terrain"], "candidates": [{"id": s} for s in source_ids]}
    context["terrain"]["routes"].append({"id": "route_test"})
    relation = {"from": layout["buildings"][0]["id"], "to": "route_test", "relation": "access"}
    layout["relations"].append(relation)
    assert validate_plan_stage("plan_layout", layout, context) == []
    relation["to"] = "missing_route"
    assert "unknown relation endpoint" in validate_plan_stage("plan_layout", layout, context)
