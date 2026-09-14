from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest
from PIL import Image

from ancientplan.cuboid3d import pipeline as p


EXAMPLE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "cuboid3d"
SCENE = "scene_01_compound"


def camera():
    return p.Camera(1200, 1000, 42, 0, 80, 0.88)


@pytest.mark.parametrize("azimuth", [0, 15, -30])
def test_ground_projection_round_trip(azimuth):
    c = camera()
    c.azimuth_deg = azimuth
    for x, y in [(0, 0), (1.25, 4), (-3, 2.5)]:
        assert c.ground_from_pixel(c.project((x, y, 0))) == pytest.approx((x, y))


@pytest.mark.parametrize("clock_angle,yaw", [(0, 0), (90, -90), (180, 180), (270, 90)])
def test_legacy_directed_long_axis(clock_angle, yaw):
    assert p.orientation_yaw_deg({"top_view_angle_deg_clockwise_from_right": clock_angle}) == yaw


def test_manual_house_fallback_rejected():
    with pytest.raises(ValueError, match="fallback is disabled"):
        p.parse_house_observations({"houses": [["OldManualHouse"]]}, camera())


def test_fixture_priors_never_supply_house_coordinates():
    scenes = json.loads((EXAMPLE / "scene_priors.json").read_text())["scenes"]
    assert len(scenes) == 3
    assert all("houses" not in scene for scene in scenes.values())
    assert not hasattr(p, "SCENES")


def test_cache_can_be_renamed_without_scene_specific_code(tmp_path):
    alias = "arbitrary_scene_name"
    for folder in ("house_detection", "building_orientation"):
        (tmp_path / folder).mkdir()
        for source in (EXAMPLE / folder).glob(SCENE + "*"):
            shutil.copyfile(source, tmp_path / folder / source.name.replace(SCENE, alias))
    image = EXAMPLE / "inputs" / (SCENE + ".png")
    with Image.open(image) as original:
        size = original.size
    detection, _ = p.load_automatic_house_instances(alias, size, tmp_path / "house_detection")
    orientation, _ = p.load_automatic_building_orientations(
        alias, image, size, tmp_path / "house_detection", tmp_path / "building_orientation"
    )
    assert len(p.parse_house_observations({}, camera(), detection, orientation)) == 4


def test_tampered_detector_binding_rejected(tmp_path):
    image = EXAMPLE / "inputs" / (SCENE + ".png")
    original = EXAMPLE / "house_detection" / (SCENE + "_building_instances.json")
    (tmp_path / original.name).write_bytes(original.read_bytes() + b"\n")
    with Image.open(image) as im:
        with pytest.raises(ValueError, match="detector-document hash"):
            p.load_automatic_building_orientations(
                SCENE, image, im.size, tmp_path, EXAMPLE / "building_orientation"
            )


def test_unrelated_input_rejected(tmp_path):
    image = tmp_path / "changed.png"
    Image.new("RGB", (1298, 1292)).save(image)
    with pytest.raises(ValueError, match="stale|size"):
        p.load_automatic_building_orientations(
            SCENE,
            image,
            (1298, 1292),
            EXAMPLE / "house_detection",
            EXAMPLE / "building_orientation",
        )


def test_orientation_identity_must_match_detection():
    detection = {"instances": [{"id": "A"}]}
    with pytest.raises(ValueError, match="exactly match"):
        p.parse_house_observations({}, camera(), detection, {"orientations": [{"id": "B"}]})


def test_geometry_solve_does_not_mutate_observations():
    obs = [p.HouseObservation("A", (450, 650), 200, 65, 40, 25, "front")]
    before = deepcopy(obs)
    house = p.solve_houses(obs, camera(), enforce_placement_constraints=False)[0]
    assert house.length > house.depth > 0
    assert house.center[2] == 0
    assert obs == before


def test_replay_imports_no_local_model_framework():
    assert not {"torch", "transformers", "ultralytics"}.intersection(sys.modules)
