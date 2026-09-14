"""Archive replay previews outside the PNG-only public showcase."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--examples", type=Path, default=Path("artifacts/replay-previews"))
    args = parser.parse_args()
    if args.examples.resolve() == (Path(__file__).resolve().parents[1] / "examples"):
        raise ValueError(
            "examples/ is reserved for the ten 2D PNG cards; choose an artifact directory"
        )
    for name in ("cuboid3d", "foundation2d"):
        source = args.replay / name
        destination = args.examples / name / "preview"
        if destination.exists():
            raise ValueError(f"Preview destination exists; review a new destination: {destination}")
        if not (source / "replay_report.json").is_file():
            raise ValueError(f"Missing completed replay report: {name}")
    for name in ("cuboid3d", "foundation2d"):
        shutil.copytree(args.replay / name, args.examples / name / "preview")
    print("Copied reviewed replay artifacts; no original inference results were changed.")


if __name__ == "__main__":
    main()
