#!/usr/bin/env python3
"""Validate generated artifacts against the handoff contracts and hard gates."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from PIL import Image, ImageChops
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent


def check_glb(path: Path) -> None:
    data = path.read_bytes()
    assert len(data) >= 20, f"{path}: truncated GLB"
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF", f"{path}: bad GLB magic"
    assert version == 2, f"{path}: expected GLB v2"
    assert declared_length == len(data), f"{path}: GLB byteLength mismatch"
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON", f"{path}: first chunk is not JSON"
    gltf = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf.get("meshes"), f"{path}: no meshes"
    assert gltf.get("nodes"), f"{path}: no nodes"


def validate_scene(
    scene_dir: Path, contract: dict, scene_validator: Draft202012Validator
) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for name in contract["per_scene_required_files"]:
        require((scene_dir / name).is_file(), f"missing {name}")
    if errors:
        return errors

    for name in contract["per_scene_required_files"]:
        path = scene_dir / name
        require(path.stat().st_size > 0, f"empty {name}")

    image_names = [name for name in contract["per_scene_required_files"] if name.endswith(".png")]
    for name in image_names:
        try:
            with Image.open(scene_dir / name) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"invalid image {name}: {exc}")

    try:
        scene = json.loads((scene_dir / "scene.json").read_text())
        schema_errors = sorted(scene_validator.iter_errors(scene), key=lambda e: list(e.path))
        errors.extend(f"scene schema: {error.message}" for error in schema_errors)
        require(
            scene["camera"]["projection"] == "orthographic", "original camera is not orthographic"
        )
        require(scene["camera"]["roll_deg"] == 0, "original camera roll is nonzero")
        require(scene["top_view_camera"]["elevation_deg"] == 90, "top view is not 90 degrees")
        require(
            scene["top_view_camera"]["projection"] == "orthographic", "top view is not orthographic"
        )
        for house in scene["houses"]:
            require(house["length"] > house["depth"], f"{house['id']}: length <= depth")
            require(house["center"][2] == 0, f"{house['id']}: z != 0")
            require(
                house["pitch_deg"] == 0 and house["roll_deg"] == 0, f"{house['id']}: tilted cuboid"
            )
            require(house["dimensions_locked"] is True, f"{house['id']}: dimensions not locked")
            visible = set(house["visible_faces"])
            require(
                not ({"front", "back"} <= visible), f"{house['id']}: front and back both visible"
            )
            require(
                not ({"left", "right"} <= visible), f"{house['id']}: left and right both visible"
            )
            evidence = house.get("evidence", {})
            solved_ratio = evidence.get("solved_length_depth_ratio_before_placement")
            final_ratio = house["length"] / house["depth"]
            require(
                isinstance(solved_ratio, (int, float))
                and math.isclose(final_ratio, solved_ratio, rel_tol=1e-9, abs_tol=1e-9),
                f"{house['id']}: final footprint ratio drifted after dimension inference",
            )
            require(
                math.isclose(
                    evidence.get("final_length_depth_ratio", float("nan")),
                    final_ratio,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"{house['id']}: recorded final footprint ratio is stale",
            )
            require(
                evidence.get("uniform_scale_preserved_ratio") is True,
                f"{house['id']}: physical body correction changed footprint ratio",
            )
            require(
                0 < float(evidence.get("prelock_uniform_body_scale", 0)) <= 1.0,
                f"{house['id']}: invalid pre-lock body scale",
            )
            depth_source = evidence.get("depth_source")
            require(
                depth_source
                in {
                    "vlm_visible_short_end_ground_corners",
                    "sam_silhouette_fixed_yaw_ratio_fit",
                    "fallback_3_to_1_unresolved_short_end",
                },
                f"{house['id']}: unaudited depth source {depth_source!r}",
            )
            if depth_source == "sam_silhouette_fixed_yaw_ratio_fit":
                require(
                    evidence.get("mask_ratio_fit", {}).get("accepted") is True,
                    f"{house['id']}: rejected SAM ratio fit was used",
                )
            if depth_source == "fallback_3_to_1_unresolved_short_end":
                require(
                    math.isclose(float(solved_ratio), 3.0, abs_tol=1e-9),
                    f"{house['id']}: unresolved depth did not use the specified 3:1 fallback",
                )
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"invalid scene.json: {exc}")

    try:
        metrics = json.loads((scene_dir / "metrics.json").read_text())
        require(metrics["accepted"] is True, "metrics did not accept scene")
        require(metrics["parsing_gate"]["passed"] is True, "parsing gate failed")
        require(
            metrics["parsing_gate"].get("source")
            == "automatic_detector_classifier_sam_plus_independent_vlm_orientation",
            "parsing gate did not use automatic house detection plus independent orientation",
        )
        require(
            metrics["parsing_gate"].get("automatic_detected_instances")
            == metrics["parsing_gate"].get("whole_house_instances"),
            "automatic/cuboid house counts differ",
        )
        require(
            metrics["parsing_gate"].get("automatic_oriented_instances")
            == metrics["parsing_gate"].get("whole_house_instances"),
            "orientation/cuboid house counts differ",
        )
        require(metrics["geometry_gate"]["passed"] is True, "geometry gate failed")
        require(metrics["hard_constraints"]["count"] == 0, "hard-constraint violations present")
        require(
            metrics["fit"]["house_semantic_silhouette_iou"] >= 0.99, "semantic fit IoU below 0.99"
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"invalid metrics.json: {exc}")

    try:
        raw = json.loads((scene_dir / "raw_semantic.json").read_text())
        detector_gate = raw["automatic_whole_house_parsing"]
        require(detector_gate["count_matches"] is True, "raw semantic detector count mismatch")
        require(
            detector_gate.get("geometry_observations_derived_from_detection") is True,
            "cuboid observations were not derived from automatic detection",
        )
        require(detector_gate["detected_count"] > 0, "raw semantic has no detected houses")
        require(bool(detector_gate["models"]), "raw semantic lacks detector model provenance")
        require(
            "Qwen/Qwen3-VL-8B-Instruct"
            in detector_gate["models"].get("semantic_proposal_models", []),
            "raw semantic did not use Qwen3-VL-8B semantic proposals",
        )
        require(
            detector_gate["models"].get("independent_orientation_model")
            == "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "raw semantic did not use the independent Qwen3-VL-32B orientation stage",
        )
        require(
            detector_gate.get("orientation_count_matches") is True, "orientation count mismatch"
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"invalid automatic parsing evidence: {exc}")

    try:
        geometry = Image.open(scene_dir / "geometry_consistent_semantic.png").convert("RGB")
        render = Image.open(scene_dir / "original_view_semantic_render.png").convert("RGB")
        require(geometry.size == render.size, "semantic target/render size mismatch")
        require(
            ImageChops.difference(geometry, render).getbbox() is None,
            "semantic render differs from locked target",
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"semantic comparison failed: {exc}")

    try:
        check_glb(scene_dir / "scene.glb")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    contract = json.loads((ROOT / "contracts" / "output_contract.json").read_text())
    schema = json.loads((ROOT / "contracts" / "scene_schema.json").read_text())
    validator = Draft202012Validator(schema)
    expected = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    if not expected:
        parser.error("No scene directories found; an empty output is not a passing batch")
    failures = 0
    for scene_name in expected:
        scene_dir = output_dir / scene_name
        errors = (
            validate_scene(scene_dir, contract, validator)
            if scene_dir.is_dir()
            else ["scene output directory missing"]
        )
        if errors:
            failures += 1
            print(f"FAIL {scene_name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {scene_name}")
    for name in contract["batch_required_files"] if len(expected) > 1 else []:
        if not (output_dir / name).is_file():
            failures += 1
            print(f"FAIL batch: missing {name}")
    if failures == 0:
        print(f"PASS batch ({len(expected)} scenes, all contract files and gates)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
