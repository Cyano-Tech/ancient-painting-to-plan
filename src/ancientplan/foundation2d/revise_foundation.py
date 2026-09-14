"""One explicit cloud revision; actual coordinates stay in the model result."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .advance_scene import load_prior
from .qwen_cloud import RUN_ROOT, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", required=True, type=Path)
    a = p.parse_args()
    check_meta, check, _ = load_prior(a.check)
    if check_meta["stage"] != "ground_check":
        raise ValueError("Expected ground check")
    hypothesis_path = Path(check_meta["context_source"])
    hypothesis = json.loads(hypothesis_path.read_text())["parsed"]
    hypothesis_meta = json.loads((hypothesis_path.parent / "request_meta.json").read_text())
    landmark_path = Path(hypothesis_meta["context_source"])
    context = deepcopy(json.loads(landmark_path.read_text())["parsed"])
    context["previous_hypotheses"] = hypothesis
    context["previous_check"] = check
    w, h = hypothesis_meta["image"]["transmitted_size"]
    measures = []
    for inst in hypothesis["instances"]:
        for alt in inst["alternatives"]:
            pts = alt["image_corners"]
            if not pts:
                continue
            lengths = [
                math.hypot(
                    (pts[(i + 1) % 4][0] - pts[i][0]) * w / 1000,
                    (pts[(i + 1) % 4][1] - pts[i][1]) * h / 1000,
                )
                for i in range(4)
            ]
            measures.append(
                {
                    "instance_id": inst["id"],
                    "alternative_id": alt["id"],
                    "corner_labels": "A=0 B=1 C=2 D=3",
                    "actual_input_corners": pts,
                    "AB_BC_CD_DA_lengths_in_image_pixels": lengths,
                    "warning": "Image lengths are not ground metric lengths. Verify which edge is the main front wall; prior review prose may disagree with actual index order.",
                }
            )
    context["geometric_correspondence_checks"] = measures
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = RUN_ROOT / (stamp + "_ground_revision_context")
    work.mkdir(mode=0o700)
    context_path = work / "context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    im = hypothesis_meta["image"]
    args = argparse.Namespace(
        stage="ground_hypotheses",
        image=im["source_path"],
        crop=im["crop_in_source_pixels"],
        base_url=None,
        run_label="revision",
        context=context,
        context_source=str(context_path),
        reference_images=[],
    )
    overlay = hypothesis_path.parent / "hypotheses_overlay.png"
    if overlay.exists():
        args.reference_images.append(
            {
                "image": str(overlay),
                "label": "上轮候选叠加图，可能错误。以实际标出的A/B/C/D位置与原图核对，不采信此前文字偷偷换角的说法。",
            }
        )
    code = run(args)
    (work / "manifest.json").write_text(
        json.dumps(
            {
                "source_check": str(a.check.resolve()),
                "context": str(context_path),
                "tiles": [{"label": "revision", "output_dir": args.output_dir, "exit_code": code}],
                "manual_coordinates_supplied": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
