"""Advance saved cloud results. No local vision, hardcoded detections, or retries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

from .qwen_cloud import RUN_ROOT, SafeError, model_matches, run


def load_prior(run_dir):
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "request_meta.json").read_text())
    raw = json.loads((run_dir / "result.json").read_text())
    norm = run_dir / (meta["stage"] + "_normalized.json")
    data_path = (
        norm
        if norm.exists() and meta["stage"] in {"inventory", "structure", "landmarks"}
        else run_dir / "result.json"
    )
    validated = run_dir / (meta["stage"] + "_validated.json")
    if validated.exists() and meta["stage"] in {"landmarks", "ground_hypotheses", "ground_check"}:
        validation = json.loads(validated.read_text())
        if (
            validation["source_sha256"]
            != hashlib.sha256((run_dir / "result.json").read_bytes()).hexdigest()
        ):
            raise SafeError("Revalidation record is stale")
        data_path = validated
    data = json.loads(data_path.read_text())
    if (
        raw.get("finish_reason") != "stop"
        or data.get("validation_issues")
        or not model_matches(meta["model_requested"], raw.get("model_returned"))
    ):
        raise SafeError("Previous run is incomplete or invalid: " + run_dir.name)
    return meta, data["parsed"], data_path


def advance(stage, prior_dir, label):
    meta, context, data_path = load_prior(prior_dir)
    expected = {
        "review": "inventory",
        "structure": "review",
        "audit": "structure",
        "ground_hypotheses": "landmarks",
        "ground_check": "ground_hypotheses",
    }
    if meta["stage"] != expected[stage]:
        raise SafeError("Wrong prior stage for " + stage)
    im = meta["image"]
    if hashlib.sha256(Path(im["source_path"]).read_bytes()).hexdigest() != im["source_sha256"]:
        raise SafeError("Source image changed since the prior stage; refusing stale coordinates.")
    args = argparse.Namespace(
        stage=stage,
        image=im["source_path"],
        crop=im["crop_in_source_pixels"],
        base_url=None,
        run_label=label,
        context=context,
        context_source=str(data_path),
    )
    if stage == "ground_check":
        landmark_path = Path(meta["context_source"])
        landmark_data = json.loads(landmark_path.read_text())
        landmark_data = landmark_data.get("parsed", landmark_data)
        landmarks = {
            k: landmark_data[k]
            for k in ("stage", "coordinate_system", "instances", "rejected", "unresolved")
            if k in landmark_data
        }
        args.context = {"landmarks": landmarks, "hypotheses": context, "require_corner_echo": True}
        overlay = Path(prior_dir) / "hypotheses_overlay.png"
        if overlay.exists():
            args.reference_images = [
                {
                    "image": str(overlay),
                    "label": "候选地基叠加图：仅是上一阶段模型假设，不是原画证据；坐标仍只相对图1。",
                }
            ]
    elif stage == "ground_hypotheses":
        overlay = Path(prior_dir) / "landmarks.png"
        if overlay.exists():
            args.reference_images = [
                {
                    "image": str(overlay),
                    "label": "逐点标记叠加图：均为模型输出，须对照原图核对；颜色和标签不是原画证据；坐标仍相对图1。",
                }
            ]
    try:
        code = run(args)
        result = json.loads((Path(args.output_dir) / "result.json").read_text())
        return {
            "label": label,
            "prior_dir": str(Path(prior_dir).resolve()),
            "output_dir": args.output_dir,
            "exit_code": code,
            "usage": result["usage"],
            "validation_issues": result["validation_issues"],
            "finish_reason": result["finish_reason"],
        }
    except (SafeError, OSError, ValueError) as exc:
        return {
            "label": label,
            "prior_dir": str(prior_dir),
            "output_dir": getattr(args, "output_dir", None),
            "exit_code": 1,
            "error_type": type(exc).__name__,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["review", "structure", "audit", "ground_hypotheses", "ground_check"]
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prior", type=Path)
    group.add_argument(
        "--manifest", type=Path, help="Advance all tiles in a prior inventory/review manifest"
    )
    p.add_argument("--label", default="region")
    args = p.parse_args()
    if args.prior:
        jobs = [(args.label, args.prior)]
    else:
        prior = json.loads(args.manifest.read_text())
        jobs = [(r.get("tile", r.get("label")), Path(r["output_dir"])) for r in prior["tiles"]]
    # Validate every input before any paid request starts.
    for _, run_dir in jobs:
        load_prior(run_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch = RUN_ROOT / (stamp + "_" + args.stage + "_batch")
    batch.mkdir(parents=True, mode=0o700)
    manifest = {
        "stage": args.stage,
        "max_parallel_requests": 2,
        "automatic_retries": 0,
        "global_instance_deduplication_done": False,
        "semantic_coverage_verified": False,
        "tiles": [],
    }
    print("Batch:", batch, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(lambda job: advance(args.stage, job[1], job[0]), jobs):
            manifest["tiles"].append(record)
            (batch / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if all(r["exit_code"] == 0 for r in manifest["tiles"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
