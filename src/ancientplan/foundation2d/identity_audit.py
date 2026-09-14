"""Two-pass, CPU/cloud-only identity audit; no scene-specific geometry rules.

Pass 1 sees just a context-expanded original crop, with no previous candidates.
Pass 2 associates that independent census with ID-only boxes. Observed geometry
is never derived from the plan's support zones or collision solver.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .complete_plan import accepted, map_point, read, request, save, stamp
from .qwen_cloud import SafeError
from .render_complete_plan import render_focus


def audit_crop(objects, size, margin=45):
    bounds = [
        min(b["source_bbox"][0] for b in objects),
        min(b["source_bbox"][1] for b in objects),
        max(b["source_bbox"][2] for b in objects),
        max(b["source_bbox"][3] for b in objects),
    ]
    # Whole vertical support chain, not just the existing roof crop. This is a
    # context margin, not an assumed building depth or a correction coordinate.
    px = max(margin, (bounds[2] - bounds[0]) * 0.25)
    py = max(margin, (bounds[3] - bounds[1]) * 0.5)
    return [
        math.floor(max(0, bounds[0] - px) * size[0] / 1000),
        math.floor(max(0, bounds[1] - py) * size[1] / 1000),
        math.ceil(min(1000, bounds[2] + px) * size[0] / 1000),
        math.ceil(min(1000, bounds[3] + py) * size[1] / 1000),
    ]


def audit(plan_path, groups):
    plan = read(plan_path)
    source = plan["source"]
    size = source["size"]
    if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
        raise SafeError("Source image changed")
    selected = [i for group in groups for i in group]
    known = {b["id"]: b for b in plan["layout"]["buildings"]}
    if len(set(selected)) != len(selected) or not set(selected) <= known.keys():
        raise SafeError("Unknown or repeated audit ID")
    folder = stamp("independent_identity_audit")
    manifest = {
        "source": source,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "jobs": [],
        "evidence": [],
        "gpu_used": False,
        "method": "blind census then ID association",
    }
    save(folder / "manifest.json", manifest)
    print("Audit manifest:", folder / "manifest.json", flush=True)

    def job(entry):
        index, ids = entry
        objects = [known[i] for i in ids]
        crop = audit_crop(objects, size)
        blind = request("plan_buildings", source["path"], crop, label=f"blind_{index}")
        census = accepted(blind["run"])

        def local(p):
            return [
                round((p[0] * size[0] / 1000 - crop[0]) * 1000 / (crop[2] - crop[0]), 3),
                round((p[1] * size[1] / 1000 - crop[1]) * 1000 / (crop[3] - crop[1]), 3),
            ]

        candidates = [
            {"id": b["id"], "bbox": local(b["source_bbox"][:2]) + local(b["source_bbox"][2:])}
            for b in objects
        ]
        focus = render_focus(plan, ids, folder / f"ids_{index}.png", crop=crop)
        context = {
            "identity_audit": True,
            "candidates": candidates,
            "blind_inventory": census,
            "blind_run": blind["run"],
            "instruction_note": "旧框只是ID定位，未提供旧名称、地基或俯视坐标。reject的几何字段可按旧bbox给占位四角，后续不会使用。missing仅可给有独立结构证据的遗漏者；裁片边缘新对象不得猜为完整实例。",
        }
        result = request(
            "plan_details",
            source["path"],
            crop,
            context,
            label=f"association_{index}",
            references=[
                {"image": str(focus), "label": "同一原画裁片的旧ID彩框，仅定位；框本身也可能错误。"}
            ],
        )
        result["candidate_ids"] = ids
        result["blind_run"] = blind["run"]
        data = deepcopy(accepted(result["run"]))
        for b in data["decisions"] + data["missing"]:
            b["bbox"] = map_point(b["bbox"][:2], crop, size) + map_point(b["bbox"][2:], crop, size)
            b["anchor"] = map_point(b["anchor"], crop, size)
            b["footprint_image"] = [map_point(p, crop, size) for p in b["footprint_image"]]
        return result, {
            "run": result["run"],
            "blind_run": blind["run"],
            "coordinate_system": "full_source_1000",
            **data,
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(job, entry): entry for entry in enumerate(groups)}
        for future in as_completed(futures):
            try:
                result, evidence = future.result()
                manifest["jobs"].append(result)
                manifest["evidence"].append(evidence)
            except (SafeError, ValueError, OSError) as exc:
                index, ids = futures[future]
                manifest["jobs"].append(
                    {
                        "exit_code": 1,
                        "candidate_ids": ids,
                        "group_index": index,
                        "error_type": type(exc).__name__,
                    }
                )
            save(folder / "manifest.json", manifest)
    print(folder / "manifest.json", flush=True)
    return folder


def collect(plan_path, runs, selections=None):
    """Recover valid independent jobs without reusing failed neighbors/results."""
    plan = read(plan_path)
    source = plan["source"]
    known = {b["id"]: b for b in plan["layout"]["buildings"]}
    manifest = {
        "source": source,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "jobs": [],
        "evidence": [],
        "method": "explicit accepted image-only evidence selection",
        "gpu_used": False,
    }
    seen = set()
    for run in runs:
        run = Path(run).resolve()
        data = deepcopy(accepted(run))
        meta = read(run / "request_meta.json")
        context = read(meta["context_source"])
        if not context.get("identity_audit") or meta["image"]["source_sha256"] != source["sha256"]:
            raise SafeError("Unrelated or non-independent audit")
        crop = meta["image"]["crop_in_source_pixels"]
        selected = set((selections or {}).get(str(run), [c["id"] for c in context["candidates"]]))
        if not selected <= {c["id"] for c in context["candidates"]}:
            raise SafeError("Selected ID absent from raw audit")
        for c in context["candidates"]:
            if c["id"] not in selected:
                continue
            if c["id"] in seen or c["id"] not in known:
                raise SafeError("Ambiguous/unknown audit ID")
            observed = map_point(c["bbox"][:2], crop, source["size"]) + map_point(
                c["bbox"][2:], crop, source["size"]
            )
            if any(abs(a - b) > 0.01 for a, b in zip(observed, known[c["id"]]["source_bbox"])):
                raise SafeError("Audited source box belongs to a different snapshot")
            seen.add(c["id"])
        data["decisions"] = [d for d in data["decisions"] if d["id"] in selected]
        for d in data["decisions"] + data["missing"]:
            d["bbox"] = map_point(d["bbox"][:2], crop, source["size"]) + map_point(
                d["bbox"][2:], crop, source["size"]
            )
            d["anchor"] = map_point(d["anchor"], crop, source["size"])
            d["footprint_image"] = [
                map_point(p, crop, source["size"]) for p in d["footprint_image"]
            ]
        manifest["jobs"].append(
            {"run": str(run), "exit_code": 0, "crop": crop, "selected_ids": sorted(selected)}
        )
        manifest["evidence"].append(
            {"run": str(run), "coordinate_system": "full_source_1000", **data}
        )
    folder = stamp("selected_identity_evidence")
    save(folder / "manifest.json", manifest)
    print(folder / "manifest.json", flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Optional run-path to selected IDs JSON; values are explicit evidence choices, never geometry overrides",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--group", action="append", nargs="+")
    mode.add_argument("--collect-runs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.collect_runs:
        collect(args.plan, args.collect_runs, read(args.selection) if args.selection else None)
    else:
        audit(args.plan, args.group)
