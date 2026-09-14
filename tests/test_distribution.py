"""Distribution and example regressions; assertions are not image accuracy scores."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from ancientplan.foundation2d.render_complete_plan import render, text, geometry_checks
from ancientplan.foundation2d.grounding import rasterize
from scripts.replay_examples import empty_output


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tests/fixtures/foundation2d"


def test_showcase_contains_only_ten_distinct_4k_pngs():
    """The public-facing showcase is PNG-only; replay data lives in fixtures."""
    from PIL import Image

    sources = json.loads((ROOT / "docs/example_sources.json").read_text())["examples"]
    assert len(sources) == len({row["met_object_id"] for row in sources}) == 10
    expected = {f"{row['sequence']:02}_{row['slug']}.png" for row in sources}
    entries = list((ROOT / "examples").iterdir())
    assert {p.name for p in entries} == expected
    assert all(p.is_file() and not p.is_symlink() for p in entries)
    assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in entries}) == 10
    for path in entries:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.size == (3840, 2160)
            image.verify()


def test_cannot_overwrite_user_output(tmp_path):
    (tmp_path / "important.txt").write_text("keep this")
    with pytest.raises(ValueError, match="empty output"):
        empty_output(tmp_path)
    assert (tmp_path / "important.txt").read_text() == "keep this"


def test_cannot_replace_file_with_output_directory(tmp_path):
    output = tmp_path / "file.txt"
    output.write_text("keep")
    with pytest.raises(ValueError):
        empty_output(output)
    assert output.read_text() == "keep"


def test_curated_plan_regression():
    plan = json.loads((EXAMPLE / "plan.json").read_text())
    before = deepcopy(plan)
    buildings = plan["layout"]["buildings"]
    assert len(buildings) == 23
    assert sum(b["category"] == "house" for b in buildings) == 20
    assert {b["id"] for b in buildings if b.get("grouped")} == {"H02", "H07", "H21"}
    assert not {"H13", "H14"}.intersection(b["id"] for b in buildings)
    assert [c["id"] for c in plan["pending_candidates"]] == ["H29"]
    assert len([b for b in buildings if b.get("observation_lock")]) == 8
    source_ids = [sid for b in buildings for sid in b["source_ids"]]
    source_ids += [r["source_id"] for r in plan["layout"]["rejected"]]
    assert len(source_ids) == len(set(source_ids)) == 35
    assert geometry_checks(plan) == []
    assert plan == before


def test_example_image_hash():
    plan = json.loads((EXAMPLE / "plan.json").read_text())
    assert (
        hashlib.sha256((EXAMPLE / plan["source"]["path"]).read_bytes()).hexdigest()
        == plan["source"]["sha256"]
    )


def test_relocated_plan_renders_from_any_working_directory(tmp_path, monkeypatch):
    folder = tmp_path / "new checkout" / "example"
    folder.mkdir(parents=True)
    shutil.copyfile(EXAMPLE / "plan.json", folder / "plan.json")
    shutil.copyfile(EXAMPLE / "source.jpg", folder / "source.jpg")
    monkeypatch.chdir(tmp_path)
    before = (folder / "plan.json").read_bytes()
    render(folder / "plan.json", png=False)
    assert (folder / "index.html").is_file()
    ET.parse(folder / "topview.svg")
    ET.parse(folder / "source_overlay.svg")
    assert (folder / "plan.json").read_bytes() == before
    assert str(tmp_path) not in (folder / "index.html").read_text()


def test_halo_text_has_separate_readable_foreground():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + text(
            5, 15, "H14 & <test>", extra='paint-order="stroke" stroke="#ffffff" stroke-width="3"'
        )
        + "</svg>"
    )
    nodes = list(ET.fromstring(svg))
    assert len(nodes) == 2
    assert nodes[1].attrib.get("stroke") is None
    assert nodes[1].text == "H14 & <test>"


def test_cpu_png_export(tmp_path):
    from PIL import Image

    path = tmp_path / "test.png"
    rasterize(
        '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8"><rect width="12" height="8" fill="#ff0000"/></svg>',
        path,
    )
    with Image.open(path) as im:
        assert im.size == (12, 8)
        assert im.convert("RGB").getpixel((4, 4)) == (255, 0, 0)
