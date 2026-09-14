"""Offline checks for provenance, facing conventions and collision detection."""

from copy import deepcopy
import unittest
from ancientplan.foundation2d.complete_plan import map_point
from ancientplan.foundation2d.plan_schema import (
    validate_building,
    validate_plan_stage,
    validate_pending,
)
from ancientplan.foundation2d.render_complete_plan import (
    rect_corners,
    contains,
    overlap,
    geometry_checks,
)
from ancientplan.foundation2d.optimize_plan import optimize
from ancientplan.foundation2d.apply_shape_evidence import constrain
from ancientplan.foundation2d.apply_identity_evidence import (
    preserve_observation_locks,
    revise,
    OBSERVED_FIELDS,
)
from ancientplan.foundation2d.identity_audit import audit_crop
from ancientplan.foundation2d.qwen_cloud import SafeError


def building():
    return {
        "id": "H01",
        "category": "house",
        "subtype": "house",
        "label": "A",
        "source_ids": ["nw:B01"],
        "source_bbox": [10, 10, 30, 30],
        "source_anchor": [20, 28],
        "source_footprint": [[10, 20], [30, 20], [30, 30], [10, 30]],
        "plan_center": [100, 100],
        "plan_size": [40, 20],
        "ratio_range": [1.5, 2.5],
        "front_clock": 6,
        "zone_id": "Z01",
        "confidence": "medium",
        "evidence": "wall visible",
        "assumption": "depth inferred",
    }


class CompletePlanTests(unittest.TestCase):
    def test_mapping_uses_original_crop(self):
        self.assertEqual(map_point([500, 500], [100, 200, 300, 600], [1000, 1000]), [200, 400])

    def test_facing_six_front_bottom(self):
        p = rect_corners(building())
        self.assertAlmostEqual(p[0][1], 110)
        self.assertAlmostEqual(p[1][1], 110)
        self.assertAlmostEqual(abs(p[0][0] - p[1][0]), 40)

    def test_facing_three_front_right(self):
        b = building()
        b["front_clock"] = 3
        p = rect_corners(b)
        self.assertAlmostEqual(p[0][0], 110)
        self.assertAlmostEqual(p[1][0], 110)
        self.assertAlmostEqual(abs(p[0][1] - p[1][1]), 40)

    def test_width_depth_not_bbox(self):
        b = building()
        b["source_bbox"] = [1, 1, 999, 3]
        self.assertEqual(validate_building(b), [])
        self.assertEqual(rect_corners(b), rect_corners(building()))

    def test_ratio_interval_enforced(self):
        b = building()
        b["plan_size"] = [90, 20]
        self.assertTrue(validate_building(b))

    def test_nonfinite_rejected(self):
        b = building()
        b["plan_center"][0] = float("nan")
        self.assertTrue(validate_building(b))

    def test_nonconvex_source_rejected(self):
        b = building()
        b["source_footprint"] = [[0, 0], [20, 20], [0, 20], [20, 0]]
        self.assertTrue(validate_building(b))

    def test_touching_not_collision(self):
        a = rect_corners(building())
        b = building()
        b["plan_center"][0] += 40
        self.assertFalse(overlap(a, rect_corners(b)))
        b["plan_center"][0] -= 1
        self.assertTrue(overlap(a, rect_corners(b)))

    def test_source_accounting(self):
        b = building()
        data = {"stage": "plan_layout", "buildings": [b], "rejected": [], "relations": []}
        context = {"candidates": [{"id": "nw:B01"}], "terrain": {"zones": [{"id": "Z01"}]}}
        self.assertEqual(validate_plan_stage("plan_layout", data, context), [])
        data["buildings"].append(deepcopy(b))
        self.assertTrue(validate_plan_stage("plan_layout", data, context))

    def test_missing_source_rejected(self):
        data = {"stage": "plan_layout", "buildings": [], "rejected": [], "relations": []}
        context = {"candidates": [{"id": "nw:B01"}], "terrain": {"zones": [{"id": "Z01"}]}}
        self.assertTrue(validate_plan_stage("plan_layout", data, context))

    def test_geometry_tests_and_no_mutation(self):
        poly = [[0, 0], [200, 0], [200, 200], [0, 200]]
        b = building()
        c = deepcopy(b)
        c["id"] = "H02"
        plan = {
            "layout": {"buildings": [b, c]},
            "terrain": {
                "plan_boundary": poly,
                "zones": [{"id": "Z01", "plan_polygon": poly}],
                "terrain": [],
            },
        }
        original = deepcopy(plan)
        self.assertEqual([c["kind"] for c in geometry_checks(plan)], ["building_overlap"])
        self.assertEqual(plan, original)

    def test_point_in_irregular_polygon(self):
        p = [[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]]
        self.assertTrue(contains([2, 2], p))
        self.assertFalse(contains([5, 8], p))

    def test_optimizer_only_moves_plan_centers(self):
        poly = [[0, 0], [200, 0], [200, 200], [0, 200]]
        b = building()
        c = deepcopy(b)
        c["id"] = "H02"
        c["source_ids"] = ["sw:B01"]
        c["plan_center"] = [130, 100]
        plan = {
            "layout": {"buildings": [b, c]},
            "terrain": {
                "plan_boundary": poly,
                "zones": [{"id": "Z01", "plan_polygon": poly}],
                "terrain": [],
            },
        }
        old = deepcopy(plan)
        result, report = optimize(plan, 20)
        self.assertEqual(plan, old)
        self.assertFalse(report["after"])
        for before, after in zip(old["layout"]["buildings"], result["layout"]["buildings"]):
            before.pop("plan_center")
            after = deepcopy(after)
            after.pop("plan_center")
            self.assertEqual(before, after)

    def test_optimizer_does_not_shrink_or_delete_when_infeasible(self):
        poly = [[0, 0], [200, 0], [200, 200], [0, 200]]
        zone = [[90, 90], [110, 90], [110, 110], [90, 110]]
        b = building()
        plan = {
            "layout": {"buildings": [b]},
            "terrain": {
                "plan_boundary": poly,
                "zones": [{"id": "Z01", "plan_polygon": zone}],
                "terrain": [],
            },
        }
        result, report = optimize(plan, 10)
        self.assertEqual(report["unresolved_ids"], ["H01"])
        self.assertEqual(result["layout"]["buildings"], [b])

    def test_final_review_cannot_pass_with_blockers(self):
        d = {
            "stage": "plan_final_check",
            "verdict": "ready_for_confirmation",
            "blocking_issues": [{"ids": ["H01"], "evidence": "overlap"}],
            "coverage": [],
            "uncertainties": [],
        }
        self.assertTrue(validate_plan_stage("plan_final_check", d, {}))
        d["blocking_issues"] = []
        self.assertEqual(validate_plan_stage("plan_final_check", d, {}), [])

    def test_metadata_repair_requires_exact_ownership(self):
        context = {"expected_source_ids": ["nw:B01"], "layout": {"buildings": [building()]}}
        d = {
            "stage": "plan_accounting",
            "assignments": [{"source_id": "nw:B01", "target": "H01", "reason": "retained"}],
        }
        self.assertEqual(validate_plan_stage("plan_accounting", d, context), [])
        d["assignments"].append(d["assignments"][0])
        self.assertTrue(validate_plan_stage("plan_accounting", d, context))

    def test_layout_cannot_widen_independent_shape_interval(self):
        b = building()
        b["plan_size"] = [60, 10]
        b["ratio_range"] = [5, 7]
        original = deepcopy(b)
        evidence = {
            "H01": {
                "status": "keep",
                "ratio_range": [1.5, 2.5],
                "source_run": "offline",
                "evidence": "sidewall",
            }
        }
        plan = {"layout": {"buildings": [b]}}
        updated, report = constrain(plan, evidence)
        self.assertEqual(b, original)
        self.assertEqual(updated["layout"]["buildings"][0]["plan_size"], [60, 30])
        self.assertTrue(report["changes"])

    def test_shape_rule_preserves_valid_aspect_and_group_envelopes(self):
        b = building()
        e = {
            "H01": {
                "status": "keep",
                "ratio_range": [1.5, 2.5],
                "source_run": "offline",
                "evidence": "wall",
            }
        }
        updated, _ = constrain({"layout": {"buildings": [b]}}, e)
        self.assertEqual(updated["layout"]["buildings"][0]["plan_size"], b["plan_size"])
        b["grouped"] = True
        b["plan_size"] = [80, 10]
        updated, _ = constrain({"layout": {"buildings": [b]}}, e)
        self.assertEqual(updated["layout"]["buildings"][0], b)

    def test_context_crop_extends_below_old_roof_box(self):
        b = building()
        b["source_bbox"] = [400, 400, 500, 500]
        self.assertEqual(audit_crop([b], [2000, 1000]), [710, 350, 1090, 550])

    def test_observation_lock_survives_reviewer_omission(self):
        b = building()
        b["observation_lock"] = {"fields": {k: deepcopy(b[k]) for k in OBSERVED_FIELDS}}
        old = {"layout": {"buildings": [b]}}
        new = deepcopy(old)
        del new["layout"]["buildings"][0]["observation_lock"]
        new["layout"]["buildings"][0]["plan_center"] = [50, 90]
        preserve_observation_locks(old, new)
        self.assertEqual(new["layout"]["buildings"][0]["observation_lock"], b["observation_lock"])

    def test_layout_cannot_move_source_to_fit_zone(self):
        b = building()
        b["observation_lock"] = {"fields": {k: deepcopy(b[k]) for k in OBSERVED_FIELDS}}
        old = {"layout": {"buildings": [b]}}
        new = deepcopy(old)
        new["layout"]["buildings"][0]["source_anchor"][1] -= 2
        with self.assertRaises(SafeError):
            preserve_observation_locks(old, new)
        self.assertTrue(validate_building(new["layout"]["buildings"][0]))

    def test_reflection_cannot_be_retained_or_merged(self):
        e = {
            "id": "H01",
            "status": "keep",
            "merge_into": None,
            "bbox": [10, 10, 30, 30],
            "anchor": [20, 20],
            "footprint_image": [[10, 10], [30, 10], [30, 30], [10, 30]],
            "ratio_range": [1, 2],
            "foundation_trace": "inverted roof beneath wall",
            "bbox_reason": "mirror below bank",
            "ground_contact": "inferred",
            "independence": "independent",
            "appearance_kind": "reflection",
        }
        context = {"identity_audit": True, "candidates": [{"id": "H01"}]}
        data = {"stage": "plan_details", "decisions": [e], "missing": []}
        self.assertTrue(validate_plan_stage("plan_details", data, context))
        e.update(status="reject", independence="nonbuilding")
        self.assertEqual(validate_plan_stage("plan_details", data, context), [])

    def test_reject_has_no_new_footprint_and_keeps_source_accounting(self):
        b = building()
        plan = {
            "source": {"sha256": "test"},
            "layout": {"buildings": [b], "rejected": [], "relations": []},
        }
        evidence = {
            "H01": {
                "status": "reject",
                "merge_into": None,
                "evidence": "water reflection",
                "foundation_trace": "roof below wall",
                "source_run": "offline",
            }
        }
        output, report = revise(plan, evidence)
        self.assertEqual(output["layout"]["buildings"], [])
        self.assertEqual(
            output["layout"]["rejected"], [{"source_id": "nw:B01", "reason": "water reflection"}]
        )
        self.assertEqual(plan["layout"]["buildings"], [b])
        self.assertFalse(report["plan_centers_changed"])

    def test_anchor_derived_from_ground_not_roof_and_source_locked(self):
        b = building()
        plan = {
            "source": {"sha256": "test"},
            "layout": {"buildings": [b], "rejected": [], "relations": []},
        }
        evidence = {
            "H01": {
                "id": "H01",
                "status": "keep",
                "bbox": [10, 10, 30, 30],
                "anchor": [20, 10],
                "footprint_image": b["source_footprint"],
                "category": "house",
                "subtype": "house",
                "label": "A",
                "front_clock": 6,
                "ratio_range": [1.5, 2.5],
                "evidence": "walls",
                "assumption": "hidden corner",
                "foundation_trace": "roof-wall-base",
                "independence": "independent",
                "ground_contact": "inferred",
                "bbox_reason": "wall extent",
                "source_run": "offline",
                "appearance_kind": "physical",
            }
        }
        output, _ = revise(plan, evidence)
        updated = output["layout"]["buildings"][0]
        self.assertEqual(updated["source_anchor"], [20, 25])
        self.assertEqual(updated["reported_source_anchor"], [20, 10])
        self.assertEqual(updated["plan_center"], b["plan_center"])
        self.assertEqual(validate_building(updated), [])

    def test_unresolved_physical_unit_does_not_claim_one_building(self):
        b = building()
        plan = {
            "source": {"sha256": "test"},
            "layout": {"buildings": [b], "rejected": [], "relations": []},
        }
        e = {
            "id": "H01",
            "status": "uncertain",
            "bbox": b["source_bbox"],
            "anchor": [20, 25],
            "footprint_image": b["source_footprint"],
            "category": "house",
            "subtype": "compound",
            "front_clock": 6,
            "ratio_range": [1, 3],
            "evidence": "attached roofs",
            "assumption": "units unresolved",
            "foundation_trace": "continuous walls unclear",
            "independence": "unresolved",
            "ground_contact": "inferred",
            "bbox_reason": "all visible walls",
            "source_run": "offline",
            "appearance_kind": "physical",
        }
        output, _ = revise(plan, {"H01": e})
        self.assertTrue(output["layout"]["buildings"][0]["grouped"])

    def test_pending_candidate_cannot_claim_foundation(self):
        c = {
            "id": "U01",
            "source_bbox": [1, 1, 20, 20],
            "plan_marker": [100, 100],
            "candidate_only": True,
            "marker_is_foundation": False,
            "label": "candidate",
            "evidence": "roof or wall",
            "uncertainty": "unresolved",
            "source_run": "offline",
        }
        plan = {"layout": {"buildings": [building()]}, "pending_candidates": [c]}
        self.assertEqual(validate_pending(plan), [])
        c["plan_size"] = [10, 10]
        self.assertTrue(validate_pending(plan))

    def test_pending_id_cannot_duplicate_retained_identity(self):
        c = {
            "id": "H01",
            "source_bbox": [1, 1, 20, 20],
            "plan_marker": [100, 100],
            "candidate_only": True,
            "marker_is_foundation": False,
            "label": "candidate",
            "evidence": "roof or wall",
            "uncertainty": "unresolved",
            "source_run": "offline",
        }
        self.assertTrue(
            validate_pending({"layout": {"buildings": [building()]}, "pending_candidates": [c]})
        )

    def test_final_counts_must_match_current_snapshot_not_history(self):
        d = {
            "stage": "plan_final_check",
            "verdict": "ready_for_confirmation",
            "blocking_issues": [],
            "coverage": [],
            "uncertainties": [],
            "verified_counts": {"active_footprint_regions": 5, "pending_candidates": 1},
        }
        context = {"geometric_facts": {"active_footprint_regions": 3, "pending_candidates": 1}}
        self.assertTrue(validate_plan_stage("plan_final_check", d, context))
        d["verified_counts"]["active_footprint_regions"] = 3
        self.assertEqual(validate_plan_stage("plan_final_check", d, context), [])


if __name__ == "__main__":
    unittest.main()
