"""Offline tests of semantic contracts and conditional projection math."""

from copy import deepcopy
import math
import unittest

from ancientplan.foundation2d.grounding import (
    convex_quad,
    expanded_crop,
    normalize_landmark_roles,
    remap_bbox,
    validate_landmarks,
    validate_hypotheses,
    validate_ground_check,
)
from ancientplan.foundation2d.plane_geometry import plane_mapping, transform, hypothesis_grid


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {"candidates": [{"id": "R1"}, {"id": "R2"}]}
        self.landmarks = {
            "instances": [
                {
                    "id": "G1",
                    "source_ids": ["R1", "R2"],
                    "identity_status": "supported",
                    "ground_status": "visible_partial",
                    "landmarks": [
                        {
                            "id": "L1",
                            "xy": [100, 500],
                            "feature": "wall_ground_corner",
                            "level": "ground",
                        },
                        {
                            "id": "L2",
                            "xy": [500, 500],
                            "feature": "wall_occluder_contact",
                            "level": "occlusion",
                        },
                    ],
                    "segments": [
                        {"id": "S1", "from": "L1", "to": "L2", "role": "occlusion_boundary"}
                    ],
                }
            ],
            "rejected": [],
        }
        self.proposal = {
            "instances": [
                {
                    "id": "G1",
                    "status": "proposed",
                    "alternatives": [
                        {
                            "id": "P1",
                            "placement": "anchored",
                            "image_corners": [[100, 500], [500, 500], [500, 300], [100, 300]],
                            "vertex_status": ["visible", "inferred", "inferred", "inferred"],
                            "vertex_landmark_ids": ["L1", None, None, None],
                            "edge_status": ["inferred"] * 4,
                            "width_depth_ratio_range": [1.5, 2.5],
                            "reference_instance_ids": [],
                            "evidence_landmark_ids": ["L1", "L2"],
                        }
                    ],
                }
            ]
        }

    def test_merged_identity_accounts_for_inputs(self):
        self.assertEqual(validate_landmarks(self.landmarks, self.candidates), [])
        data = deepcopy(self.landmarks)
        data["instances"][0]["source_ids"].pop()
        self.assertTrue(validate_landmarks(data, self.candidates))

    def test_occlusion_cannot_be_promoted_to_ground(self):
        data = deepcopy(self.landmarks)
        data["instances"][0]["segments"][0]["role"] = "ground_edge"
        self.assertTrue(validate_landmarks(data, self.candidates))
        data["instances"][0]["landmarks"][1]["level"] = "ground"
        self.assertTrue(validate_landmarks(data, self.candidates))

    def test_role_alias_does_not_change_point_semantics(self):
        data = deepcopy(self.landmarks)
        data["instances"][0]["segments"][0]["role"] = "visible_ground_line"
        normalized, changes = normalize_landmark_roles(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(data["instances"][0]["landmarks"], normalized["instances"][0]["landmarks"])
        self.assertTrue(validate_landmarks(normalized, self.candidates))

    def test_anchor_must_be_exact_ground_landmark(self):
        self.assertEqual(validate_hypotheses(self.proposal, self.landmarks), [])
        data = deepcopy(self.proposal)
        data["instances"][0]["alternatives"][0]["vertex_landmark_ids"][0] = "L2"
        self.assertTrue(validate_hypotheses(data, self.landmarks))
        data = deepcopy(self.proposal)
        data["instances"][0]["alternatives"][0]["image_corners"][0][0] += 1
        self.assertTrue(validate_hypotheses(data, self.landmarks))

    def test_shape_only_has_no_placement_geometry(self):
        data = deepcopy(self.proposal)
        alt = data["instances"][0]["alternatives"][0]
        alt["placement"] = "shape_only"
        self.assertTrue(validate_hypotheses(data, self.landmarks))
        for k in ("image_corners", "vertex_status", "vertex_landmark_ids", "edge_status"):
            alt[k] = []
        self.assertEqual(validate_hypotheses(data, self.landmarks), [])

    def test_alternative_ids_are_instance_local(self):
        data = deepcopy(self.proposal)
        prior = deepcopy(self.landmarks)
        prior["instances"].append({"id": "G2", "landmarks": []})
        alt = deepcopy(data["instances"][0]["alternatives"][0])
        alt.update(
            placement="inferred", vertex_status=["inferred"] * 4, vertex_landmark_ids=[None] * 4
        )
        data["instances"].append({"id": "G2", "status": "proposed", "alternatives": [alt]})
        self.assertEqual(validate_hypotheses(data, prior), [])
        data["instances"][1]["alternatives"].append(deepcopy(alt))
        self.assertTrue(validate_hypotheses(data, prior))

    def test_quad_self_crossing_is_invalid(self):
        self.assertFalse(convex_quad([[0, 0], [10, 10], [10, 0], [0, 10]]))
        self.assertFalse(convex_quad([[0, 0], [1, 0], [2, 0], [3, 0]]))
        self.assertTrue(convex_quad([[0, 0], [10, 0], [8, 10], [2, 10]]))

    def test_point_mapping_and_crop_expansion(self):
        meta = {
            "crop_in_source_pixels": [100, 100, 300, 200],
            "source_size_after_exif": [1000, 1000],
        }
        crop = expanded_crop(meta)
        self.assertEqual(crop, [76, 85, 324, 245])
        mapped = remap_bbox([0, 0, 1000, 1000], meta["crop_in_source_pixels"], crop)
        self.assertTrue(0 < mapped[0] < mapped[2] < 1000)
        self.assertTrue(0 < mapped[1] < mapped[3] < 1000)

    def test_check_covers_every_namespaced_alternative(self):
        context = {"landmarks": self.landmarks, "hypotheses": self.proposal}
        data = {
            "decisions": [
                {"instance_id": "G1", "alternative_id": "P1", "verdict": "needs_revision"}
            ]
        }
        self.assertEqual(validate_ground_check(data, context), [])
        self.assertTrue(validate_ground_check({"decisions": []}, context))

    def test_corner_echo_prevents_silent_reordering_in_review(self):
        context = {
            "landmarks": self.landmarks,
            "hypotheses": self.proposal,
            "require_corner_echo": True,
        }
        corners = self.proposal["instances"][0]["alternatives"][0]["image_corners"]
        decision = {
            "instance_id": "G1",
            "alternative_id": "P1",
            "verdict": "usable_hypothesis",
            "evaluated_corners": deepcopy(corners),
            "front_edge_indices": [0, 1],
            "order_consistent": True,
        }
        self.assertEqual(validate_ground_check({"decisions": [decision]}, context), [])
        decision["evaluated_corners"] = corners[1:] + corners[:1]
        self.assertTrue(validate_ground_check({"decisions": [decision]}, context))
        decision["evaluated_corners"] = deepcopy(corners)
        decision["front_edge_indices"] = [1, 2]
        self.assertTrue(validate_ground_check({"decisions": [decision]}, context))
        decision["verdict"] = "needs_revision"
        decision["order_consistent"] = False
        self.assertEqual(validate_ground_check({"decisions": [decision]}, context), [])

    def test_projection_preserves_assumed_aspect_not_pixel_ratio(self):
        corners = [[100, 600], [850, 650], [700, 200], [180, 250]]
        for ratio in (1.25, 2.75):
            mapping = plane_mapping(corners, ratio)
            target = [[0, 0], [ratio, 0], [ratio, 1], [0, 1]]
            for p, q in zip(corners, target):
                actual = transform(mapping["image_to_plane"], [v / 1000 for v in p])
                self.assertLess(math.dist(actual, q), 1e-8)
                back = transform(mapping["plane_to_image"], actual)
                self.assertLess(math.dist(back, [v / 1000 for v in p]), 1e-8)
            self.assertFalse(mapping["calibration_verified"])
            lines, _ = hypothesis_grid(corners, ratio)
            self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
