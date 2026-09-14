"""Presentation transforms are cosmetic and must not change inference data."""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import render_example_cards as cards


def plan_fixture():
    root = Path(__file__).resolve().parents[1]
    return root / "tests/fixtures/foundation2d"


def test_plan_drawing_is_valid_svg_and_pure():
    plan = json.loads((plan_fixture() / "plan.json").read_text())
    before = deepcopy(plan)
    svg = '<svg xmlns="http://www.w3.org/2000/svg">' + cards.draw_plan(plan) + "</svg>"
    ET.fromstring(svg)
    bounds = cards.content_bounds(plan)
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert plan == before


def test_card_keeps_source_geometry_and_has_4k_dimensions(tmp_path, monkeypatch):
    fixture = plan_fixture()
    plan = json.loads((fixture / "plan.json").read_text())
    before = deepcopy(plan)
    observed = []
    monkeypatch.setattr(cards, "rasterize", lambda svg, output: observed.append(svg))
    row = {"sequence": 1, "title": "A < B", "theme": "Test & review", "credit": "Test source"}
    cards.card(row, plan, fixture / "source.jpg", tmp_path / "card.png")
    tree = ET.fromstring(observed[0])
    assert tree.attrib["width"] == "3840" and tree.attrib["height"] == "2160"
    assert "A &lt; B" in observed[0]
    assert plan == before


def test_wrong_image_hash_fails_before_rendering(tmp_path):
    import pytest

    plan = json.loads((plan_fixture() / "plan.json").read_text())
    image = tmp_path / "not-source.jpg"
    image.write_bytes(b"not the original")
    with pytest.raises(ValueError, match="does not match"):
        cards.card({}, plan, image, tmp_path / "card.png")


def test_close_labels_are_separated_without_moving_anchors():
    buildings = [
        {"id": "H01", "source_anchor": [500, 500]},
        {"id": "H02", "source_anchor": [500, 500]},
    ]
    before = deepcopy(buildings)
    boxes = cards.label_boxes(buildings, (0, 0, 1000, 1000))
    a, b = boxes
    assert a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
    assert buildings == before


def test_final_export_rejects_stale_review_hash(tmp_path, monkeypatch):
    import pytest

    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    run = tmp_path / "review"
    run.mkdir()
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"plan_sha256": "not-the-current-hash"}))
    (run / "request_meta.json").write_text(
        json.dumps({"stage": "plan_final_check", "context_source": str(context)})
    )
    monkeypatch.setattr(cards, "accepted", lambda _: {"verdict": "ready_for_confirmation"})
    with pytest.raises(ValueError, match="another plan snapshot"):
        cards.require_review({"ready": True, "final_check": {"run": str(run)}}, plan_path, {})
