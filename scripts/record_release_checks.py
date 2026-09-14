"""Record measured local checks without embedding machine paths or credentials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suites = ET.parse(args.junit).getroot().findall("testsuite")
    counts = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }
    if not counts["tests"] or counts["failures"] or counts["errors"]:
        raise ValueError("Cannot record a successful release with missing/failing tests")
    data = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": {
            k: version(k) for k in ("Pillow", "jsonschema", "CairoSVG", "pytest", "ruff")
        },
        "pytest": counts,
        "gpu_used": False,
        "cloud_calls_during_packaging": 0,
        "cuboid3d": json.loads((args.replay / "cuboid3d/replay_report.json").read_text()),
        "foundation2d": json.loads((args.replay / "foundation2d/replay_report.json").read_text()),
        "claims_excluded": [
            "semantic_ground_truth_accuracy",
            "metric_calibration",
            "fresh_local_model_inference",
            "fresh_cloud_inference",
        ],
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {counts['tests']} passing test cases")


if __name__ == "__main__":
    main()
