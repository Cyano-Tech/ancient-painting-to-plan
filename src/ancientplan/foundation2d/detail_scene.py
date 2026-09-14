"""Independent cloud re-detection on automatically selected ambiguous house groups.

Only crop selection uses prior boxes. The cloud sees original pixels and an
EMPTY candidate list, not prior counts, classifications, coordinates or evidence.
Output is a new local observation, not an automatic overwrite of global identity.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from .advance_scene import load_prior
from .qwen_cloud import RUN_ROOT, run
from .summarize_reviews import iou, to_source_bbox


def ambiguous_groups(objects, threshold=0.05):
    houses = [o for o in objects if o["category"] == "house"]
    neighbors = {o["id"]: set() for o in houses}
    by_id = {o["id"]: o for o in houses}
    for i, a in enumerate(houses):
        for b in houses[i + 1 :]:
            if iou(a["bbox"], b["bbox"]) >= threshold:
                neighbors[a["id"]].add(b["id"])
                neighbors[b["id"]].add(a["id"])
    seen = set()
    groups = []
    for oid in sorted(neighbors):
        if oid in seen or not neighbors[oid]:
            continue
        stack = [oid]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        seen.update(component)
        members = [by_id[i] for i in sorted(component)]
        boxes = [m["bbox"] for m in members]
        score = max(iou(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
        groups.append(
            {
                "ids": sorted(component),
                "bbox": [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ],
                "score": score,
            }
        )
    return sorted(groups, key=lambda g: (-g["score"], g["ids"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", required=True, type=Path)
    p.add_argument("--max-regions", type=int, default=3)
    p.add_argument("--padding", type=float, default=0.18)
    args = p.parse_args()
    if not 1 <= args.max_regions <= 4 or not 0 <= args.padding <= 0.5:
        raise ValueError("Bounded limits: 1-4 regions, padding 0-0.5")
    meta, data, _ = load_prior(args.prior)
    im = meta["image"]
    groups = ambiguous_groups(data["objects"])[: args.max_regions]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = RUN_ROOT / (stamp + "_independent_details")
    out.mkdir(parents=True, mode=0o700)
    empty = {"stage": "independent_detail_inventory", "objects": []}
    (out / "cloud_context.json").write_text(json.dumps(empty) + "\n")
    jobs = []
    for i, g in enumerate(groups):
        box = to_source_bbox(g["bbox"], im["crop_in_source_pixels"])
        a, b, c, d = box
        padx, pady = (c - a) * args.padding, (d - b) * args.padding
        w, h = im["source_size_after_exif"]
        crop = [
            max(0, int(a - padx)),
            max(0, int(b - pady)),
            min(w, round(c + padx)),
            min(h, round(d + pady)),
        ]
        jobs.append(
            {
                "label": f"detail_{i + 1}",
                "crop": crop,
                "trigger_ids": g["ids"],
                "overlap_score": g["score"],
            }
        )
    manifest = {
        "prior_dir": str(args.prior.resolve()),
        "selection": "connected house bbox overlap >= 0.05, padding by region size",
        "old_candidates_sent_to_cloud": False,
        "automatic_global_replacement": False,
        "max_parallel_requests": 2,
        "tiles": [],
    }
    print("Independent detail batch:", out, flush=True)

    def execute(job):
        a = argparse.Namespace(
            stage="review",
            image=im["source_path"],
            crop=job["crop"],
            base_url=None,
            run_label=job["label"],
            context=empty,
            context_source=str(out / "cloud_context.json"),
        )
        try:
            code = run(a)
            result = json.loads((Path(a.output_dir) / "result.json").read_text())
            return {
                **job,
                "output_dir": a.output_dir,
                "exit_code": code,
                "usage": result["usage"],
                "validation_issues": result["validation_issues"],
            }
        except Exception as exc:
            return {
                **job,
                "output_dir": getattr(a, "output_dir", None),
                "exit_code": 1,
                "error_type": type(exc).__name__,
            }

    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(execute, jobs):
            manifest["tiles"].append(record)
            (out / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if jobs and all(r["exit_code"] == 0 for r in manifest["tiles"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
