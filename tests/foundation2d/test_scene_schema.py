"""Offline provenance/geometry checks. No network or local vision model."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from ancientplan.foundation2d.scene_schema import validate_stage
from ancientplan.foundation2d.summarize_reviews import to_source_bbox, iou
from ancientplan.foundation2d.detail_scene import ambiguous_groups
from ancientplan.foundation2d.normalize_structure import normalize as normalize_structure
from ancientplan.foundation2d.render_glass import build_svg


class SceneSchemaTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "objects": [{"id": "H1", "category": "house"}, {"id": "H2", "category": "house"}]
        }
        self.review = {
            "objects": [
                {
                    "id": "R1",
                    "category": "house",
                    "bbox": [1, 2, 10, 20],
                    "source_ids": ["H1", "H2"],
                    "review_status": "supported",
                }
            ],
            "rejected": [],
        }
        self.structure = {
            "objects": [
                {
                    "id": "R1",
                    "category": "house",
                    "status": "proposed",
                    "visible_paths": [[[1, 2], [10, 2]]],
                    "hidden_paths": [],
                    "reference_ids": [],
                    "occluded_by": [],
                    "support_plane": None,
                    "footprint": {
                        "kind": "polygon",
                        "points": [[1, 2], [10, 2], [10, 8]],
                        "edge_status": ["visible", "inferred", "unknown"],
                    },
                }
            ],
            "support_planes": [],
        }

    def test_merge_accounts_for_both_sources(self):
        self.assertEqual(validate_stage("review", self.review, self.inventory), [])

    def test_omitted_or_rejected_retained_sources_fail(self):
        data = deepcopy(self.review)
        data["objects"][0]["source_ids"] = ["H1"]
        self.assertTrue(validate_stage("review", data, self.inventory))
        data = deepcopy(self.review)
        data["rejected"] = [{"source_id": "H1", "reason": "conflict"}]
        self.assertTrue(validate_stage("review", data, self.inventory))

    def test_structure_provenance_and_unknown(self):
        self.assertEqual(validate_stage("structure", self.structure, self.review), [])
        data = deepcopy(self.structure)
        data["objects"][0]["footprint"] = {"kind": "unknown", "points": [], "edge_status": []}
        self.assertEqual(validate_stage("structure", data, self.review), [])

    def test_missing_edge_provenance_rejected(self):
        data = deepcopy(self.structure)
        data["objects"][0]["footprint"]["edge_status"].pop()
        self.assertTrue(validate_stage("structure", data, self.review))

    def test_nan_and_bool_coordinates_rejected(self):
        for v in (float("nan"), True, 1001):
            data = deepcopy(self.structure)
            data["objects"][0]["visible_paths"][0][0][0] = v
            self.assertTrue(validate_stage("structure", data, self.review))

    def test_deferred_cannot_have_completed_footprint(self):
        data = deepcopy(self.structure)
        data["objects"][0]["status"] = "deferred"
        self.assertTrue(validate_stage("structure", data, self.review))

    def test_audit_cannot_accept_uncertain_or_unknown(self):
        data = {
            "objects": [{"id": "R1", "verdict": "needs_review", "safe_for_ground_projection": True}]
        }
        self.assertTrue(validate_stage("audit", data, self.structure))
        data["objects"][0]["verdict"] = "plausible"
        self.assertEqual(validate_stage("audit", data, self.structure), [])
        unknown = deepcopy(self.structure)
        unknown["objects"][0]["footprint"]["kind"] = "unknown"
        self.assertTrue(validate_stage("audit", data, unknown))

    def test_source_mapping_uses_crop_and_both_dimensions(self):
        self.assertEqual(to_source_bbox([0, 0, 1000, 1000], [20, 30, 220, 330]), [20, 30, 220, 330])
        self.assertEqual(
            to_source_bbox([100, 200, 500, 600], [20, 30, 220, 330]), [40, 90, 120, 210]
        )
        self.assertEqual(iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)

    def test_detail_selection_is_geometric_not_image_specific(self):
        objects = [
            {"id": "a", "category": "house", "bbox": [0, 0, 20, 20]},
            {"id": "b", "category": "house", "bbox": [10, 0, 30, 20]},
            {"id": "c", "category": "house", "bbox": [500, 500, 600, 600]},
            {"id": "t", "category": "tree", "bbox": [0, 0, 20, 20]},
        ]
        groups = ambiguous_groups(objects)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["ids"], ["a", "b"])
        self.assertEqual(groups[0]["bbox"], [0, 0, 30, 20])

    def test_string_encoded_paths_decode_without_numeric_changes(self):
        data = deepcopy(self.structure)
        data["objects"][0]["visible_paths"] = ["[[1,2],[10,2]]"]
        normalized, changes = normalize_structure(data)
        self.assertIsInstance(data["objects"][0]["visible_paths"][0], str)
        self.assertEqual(normalized["objects"][0]["visible_paths"], [[[1, 2], [10, 2]]])
        self.assertEqual(len(changes), 1)
        self.assertEqual(validate_stage("structure", normalized, self.review), [])

    def test_preview_gates_failed_audit_without_changing_geometry(self):
        with tempfile.TemporaryDirectory(prefix="scene_svg_test_") as tmp:
            directory = Path(tmp)
            (directory / "input.jpg").write_bytes(b"test-not-rasterized")
            meta = {"image": {"transmitted_size": [100, 50]}}
            before = deepcopy(self.structure)
            audit = {
                "objects": [
                    {
                        "id": "R1",
                        "verdict": "needs_review",
                        "safe_for_ground_projection": False,
                        "issues": ["check"],
                    }
                ]
            }
            svg = build_svg(directory, meta, self.structure, self.review, audit, "glass")
            ET.fromstring(svg)
            self.assertIn('class="object pending"', svg)
            self.assertIn("本次未补隐藏线", svg)
            self.assertEqual(before, self.structure)


if __name__ == "__main__":
    unittest.main()
