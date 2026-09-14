"""Offline unit tests only; no calls, real credentials, or GPU."""

from pathlib import Path
import tempfile
import unittest

from ancientplan.foundation2d.qwen_cloud import (
    SafeError,
    model_matches,
    parse_answer,
    read_key,
    redact,
    validate_base_url,
    validate_inventory,
)
from ancientplan.foundation2d.inventory_tiles import tile_boxes
from ancientplan.foundation2d.normalize_inventory import normalize, semantic_warnings


class CloudSafetyTests(unittest.TestCase):
    def test_returned_model_cannot_silently_downgrade(self):
        self.assertTrue(model_matches("qwen3.8-max", "qwen3.8-max"))
        self.assertTrue(model_matches("qwen3.8-max", "qwen3.8-max-0902"))
        self.assertFalse(model_matches("qwen3.8-max", "qwen3.8-flash"))

    def test_key_file_is_parsed_not_executed(self):
        with tempfile.TemporaryDirectory(prefix="qwen_test_") as directory:
            p = Path(directory) / "test.env"
            p.write_text("export DASHSCOPE_API_KEY='test-only-fake-value'\n", encoding="utf-8")
            p.chmod(0o600)
            self.assertEqual(read_key(p), "test-only-fake-value")
            p.chmod(0o644)
            with self.assertRaises(SafeError):
                read_key(p)

    def test_reject_missing_and_unsafe_endpoints(self):
        for url in (
            None,
            "http://example.com/v1",
            "https://key@example.com/v1",
            "https://example.com/v1?key=value",
        ):
            with self.assertRaises(SafeError):
                validate_base_url(url)

    def test_recursive_secret_redaction(self):
        self.assertEqual(
            redact({"error": ["Bearer test-only-fake-value"]}, "test-only-fake-value"),
            {"error": ["Bearer [REDACTED]"]},
        )

    def test_json_fences_and_truncation(self):
        result, issues = parse_answer('```json\n{"objects": []}\n```')
        self.assertEqual(result, {"objects": []})
        self.assertFalse(issues)
        result, issues = parse_answer('{"objects": [')
        self.assertIsNone(result)
        self.assertTrue(issues)

    def test_box_range_and_duplicate_ids(self):
        data = {
            "objects": [
                {"id": "H1", "category": "house", "bbox": [1, 2, 3, 4]},
                {"id": "H1", "category": "house", "bbox": [1, 2, 1001, 4]},
            ]
        }
        self.assertEqual(len(validate_inventory(data)), 2)

    def test_four_tiles_cover_image_bounds(self):
        boxes = tile_boxes(3946, 3716)
        self.assertEqual(len(boxes), 4)
        for x in (0, 100, 1973, 3000, 3945):
            for y in (0, 100, 1858, 3000, 3715):
                self.assertTrue(any(a <= x < c and b <= y < d for _, (a, b, c, d) in boxes))
        self.assertGreater(boxes[0][1][2], boxes[1][1][0])
        self.assertGreater(boxes[0][1][3], boxes[2][1][1])

    def test_alias_normalization_preserves_raw_geometry(self):
        raw = {
            "objects": [
                {"id": "T1", "category": "tree_group", "bbox": [1, 2, 3, 4], "grouped": True}
            ]
        }
        normalized, changes = normalize(raw)
        self.assertEqual(raw["objects"][0]["category"], "tree_group")
        self.assertEqual(normalized["objects"][0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(normalized["objects"][0]["category"], "tree")
        self.assertEqual(len(changes), 1)

    def test_conflicting_classes_are_flagged_not_rewritten(self):
        data = {
            "objects": [
                {"id": "T1", "category": "tree", "bbox": [1, 2, 3, 4]},
                {"id": "M1", "category": "mountain", "bbox": [1, 2, 3, 4]},
            ]
        }
        self.assertEqual(len(semantic_warnings(data)), 1)
        self.assertEqual(len(data["objects"]), 2)


if __name__ == "__main__":
    unittest.main()
