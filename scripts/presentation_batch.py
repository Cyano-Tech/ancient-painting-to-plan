"""Bounded, resumable cloud inference for multiple real presentation examples.

Uses existing stage schemas and prompts. No per-painting geometry corrections,
no local inference frameworks, no HTTP retries. Invalid drafts remain explicit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

from ancientplan.foundation2d.complete_plan import (
    accepted,
    apply_review,
    critique,
    draft,
    final_check,
    layout,
    read,
    request,
    save,
)
from ancientplan.foundation2d.inventory_tiles import tile_boxes
from ancientplan.foundation2d.optimize_plan import optimize
from ancientplan.foundation2d.qwen_cloud import SafeError
from ancientplan.foundation2d.render_complete_plan import render, geometry_checks

if __package__:
    from .recover_cloud_result import recover_job
else:
    from recover_cloud_result import recover_job


def select(catalog: Path, output: Path):
    """Curatorial selection affects the input set and labels, never recognition."""
    entries = {r["id"]: r for r in read(catalog)}
    choices = [
        (56, "cassia_studio", "桂荫书斋", "书斋 · 林地 · 院落"),
        (60, "willow_pavilion", "柳岸亭居", "水岸 · 亭榭 · 树根"),
        (61, "river_village", "帆影江村", "村落 · 舟行 · 两岸"),
        (59, "stream_bridge", "溪桥山居", "溪流 · 小桥 · 山居"),
        (46, "lakeshore_pavilion", "湖畔亭榭", "扇面 · 湖岸 · 支承面"),
        (34, "garden_estate", "青绿园林", "青绿山水 · 园林 · 庭院"),
        (31, "orchid_pavilion", "兰亭山水", "群山 · 楼阁 · 台地"),
        (28, "yueyang_pavilion", "岳阳楼阁", "楼阁 · 台基 · 多层结构"),
        (20, "autumn_temples", "秋山梵宇", "山寺 · 谷地 · 林木"),
        (47, "snowy_retreat", "雪江村舍", "雪景 · 村舍 · 水岸"),
    ]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Selection already exists; do not overwrite an active batch")
    selected = []
    for index, (cid, slug, title, theme) in enumerate(choices, 1):
        row = entries[cid]
        folder = output / f"{index:02}_{slug}"
        folder.mkdir()
        shutil.copyfile(row["path"], folder / "source.jpg")
        image = folder / "source.jpg"
        selected.append(
            {
                **row,
                "sequence": index,
                "slug": slug,
                "title": title,
                "theme": theme,
                "directory": folder.name,
                "path": "source.jpg",
                "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "title_is_presentation_theme": True,
            }
        )
    save(
        output / "selection.json",
        {
            "examples": selected,
            "local_gpu_used": False,
            "note": "Ten distinct real paintings selected for presentation, not random evaluation data.",
        },
    )
    print(output / "selection.json", flush=True)


def survey(folder: Path, row: dict, repair=False, correction_limit=1):
    """Full scene plus generic overlapping crops; reuse completed valid jobs."""
    manifest_path = folder / "survey.json"
    image = folder / "source.jpg"
    width, height = row["size"]
    manifest = (
        read(manifest_path)
        if manifest_path.exists()
        else {
            "source": {
                "path": str(image.resolve()),
                "size": [width, height],
                "sha256": row["source_sha256"],
            },
            "cloud_model": "qwen3.8-max",
            "gpu_used": False,
            "jobs": [],
            "batch": str(folder.resolve()),
        }
    )
    if hashlib.sha256(image.read_bytes()).hexdigest() != manifest["source"]["sha256"]:
        raise SafeError("Source changed")
    jobs = [("plan_terrain", None, "whole")] + [
        ("plan_buildings", crop, name) for name, crop in tile_boxes(width, height)
    ]
    for stage, crop, label in jobs:
        existing = next(
            (j for j in manifest["jobs"] if j["stage"] == stage and j["label"] == label), None
        )
        if existing:
            if existing["exit_code"] and repair:
                replacement = recover_job(
                    existing, allow_cloud=True, max_cloud_corrections=correction_limit
                )
                manifest["jobs"][manifest["jobs"].index(existing)] = replacement
                save(manifest_path, manifest)
                existing = replacement
            if existing["exit_code"]:
                raise SafeError(
                    f"Explicitly inspect rejected survey stage before resuming: {existing['run']}"
                )
            accepted(existing["run"])
            continue
        result = request(stage, str(image), crop=crop, label=label)
        manifest["jobs"].append(result)
        save(manifest_path, manifest)
        if result["exit_code"]:
            if repair:
                result = recover_job(
                    result, allow_cloud=True, max_cloud_corrections=correction_limit
                )
                manifest["jobs"][-1] = result
                save(manifest_path, manifest)
            if result["exit_code"]:
                raise SafeError(f"Rejected survey response: {result['run']}")
    return manifest_path


def process(root: Path, row: dict, phase: str, repair=False, correction_limit=1):
    folder = root / row["directory"]
    state_path = folder / "state.json"
    state = (
        read(state_path)
        if state_path.exists()
        else {"sequence": row["sequence"], "slug": row["slug"]}
    )
    try:
        manifest = survey(folder, row, repair, correction_limit)
        state["survey"] = str(manifest)
        save(state_path, state)
        if phase == "survey":
            return state
        if not state.get("plan"):
            result = state.get("layout_rejected") or layout(manifest)
            if isinstance(result, dict):
                state["layout_rejected"] = result
                save(state_path, state)
                if not repair:
                    raise SafeError(f"Rejected layout response: {result['run']}")
                result = recover_job(
                    result, allow_cloud=True, max_cloud_corrections=correction_limit
                )
                state["layout_rejected"] = result
                save(state_path, state)
                accepted(result["run"])
                result = draft(result["run"], manifest)
            state["plan"] = str(result / "plan.json")
            save(state_path, state)
        if not state.get("fitted_plan"):
            plan = read(state["plan"])
            fitted, report = optimize(plan, max_shift=40)
            target = folder / "fitted"
            target.mkdir(exist_ok=True)
            fitted["parent_plan"] = state["plan"]
            save(target / "plan.json", fitted)
            save(target / "fit_report.json", report)
            render(target / "plan.json", png=True)
            state["fitted_plan"] = str(target / "plan.json")
            state["geometry_issues"] = geometry_checks(fitted)
            save(state_path, state)
        if phase == "layout":
            state.pop("error", None)
            save(state_path, state)
            return state
        if state.get("independent_identity_requested") and not state.get("identity_audit"):
            from ancientplan.foundation2d.identity_audit import audit

            plan = read(state["fitted_plan"])
            ids = [b["id"] for b in plan["layout"]["buildings"]]
            # One context group, two serial calls; no extra concurrent request.
            evidence = audit(state["fitted_plan"], [ids]) / "manifest.json"
            state["identity_audit"] = str(evidence)
            save(state_path, state)
        if state.get("independent_identity_requested") and not state.get("identity_applied"):
            from ancientplan.foundation2d.apply_identity_evidence import apply as apply_identity

            revised = apply_identity(state["fitted_plan"], state["identity_audit"])
            fitted, report = optimize(read(revised / "plan.json"), max_shift=40)
            target = folder / f"identity_{len(state.get('review_history', [])) + 1:02}"
            target.mkdir(exist_ok=True)
            save(target / "plan.json", fitted)
            save(target / "fit_report.json", report)
            render(target / "plan.json", png=True)
            state["fitted_plan"] = str(target / "plan.json")
            state["identity_applied"] = True
            state["detail_evidence"] = state["identity_audit"]
            save(state_path, state)
        if not state.get("critique"):
            result = critique(
                state["fitted_plan"],
                details=state.get("detail_evidence"),
                notes=state.get("review_notes"),
            )
            state["critique"] = result
            save(state_path, state)
        if state["critique"]["exit_code"]:
            if repair:
                state["critique"] = recover_job(
                    state["critique"], allow_cloud=True, max_cloud_corrections=correction_limit
                )
                save(state_path, state)
            accepted(state["critique"]["run"])
        if not state.get("reviewed_plan"):
            result = apply_review(
                state["fitted_plan"], state["critique"]["run"], repair_accounting=repair
            )
            plan = read(result / "plan.json")
            fitted, report = optimize(plan, max_shift=40)
            round_number = len(state.get("review_history", [])) + 1
            target = folder / ("reviewed" if round_number == 1 else f"reviewed_{round_number:02}")
            target.mkdir(exist_ok=True)
            save(target / "plan.json", fitted)
            save(target / "fit_report.json", report)
            render(target / "plan.json", png=True)
            state["reviewed_plan"] = str(target / "plan.json")
            state["geometry_issues"] = geometry_checks(fitted)
            save(state_path, state)
        if not state.get("final_check"):
            state["final_check"] = final_check(state["reviewed_plan"])
            save(state_path, state)
        if state["final_check"]["exit_code"] and repair:
            state["final_check"] = recover_job(
                state["final_check"], allow_cloud=True, max_cloud_corrections=correction_limit
            )
            save(state_path, state)
        verdict = accepted(state["final_check"]["run"])
        state["ready"] = verdict["verdict"] == "ready_for_confirmation" and not any(
            g["severity"] == "major" for g in state["geometry_issues"]
        )
        state.pop("error", None)
    except (SafeError, OSError, ValueError, KeyError) as exc:
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        state["ready"] = False
    save(state_path, state)
    print(
        json.dumps(
            {"scene": row["directory"], "ready": state.get("ready"), "error": state.get("error")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return state


def next_review(root, row, observations, independent_identity=False):
    """Explicitly schedule another visual review, preserving all prior snapshots."""
    folder = root / row["directory"]
    state_path = folder / "state.json"
    state = read(state_path)
    if not state.get("reviewed_plan") or not state.get("final_check"):
        raise SafeError("Finish the preceding review before scheduling another")
    previous = accepted(state["final_check"]["run"])
    plan = read(state["reviewed_plan"])
    notes = {
        "source_sha256": plan["source"]["sha256"],
        "observations": observations,
        "previous_blocking_issues": previous["blocking_issues"],
        "instruction": "对照原画独立复核这些疑点，不将审阅意见当成真值。不得靠删除可见对象、改变真实分类含义来使检查通过。",
    }
    history = state.get("review_history", [])
    notes_path = folder / f"review_notes_{len(history) + 2:02}.json"
    save(notes_path, notes)
    snapshot = {k: v for k, v in state.items() if k != "review_history"}
    state["review_history"] = history + [snapshot]
    state["fitted_plan"] = state["reviewed_plan"]
    state["review_notes"] = str(notes_path.resolve())
    for key in (
        "reviewed_plan",
        "critique",
        "final_check",
        "error",
        "identity_audit",
        "identity_applied",
        "detail_evidence",
    ):
        state.pop(key, None)
    state["independent_identity_requested"] = independent_identity
    state["ready"] = False
    save(state_path, state)
    print("Scheduled next review:", row["directory"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=["survey", "layout", "review"], default="layout")
    p.add_argument("--ids", nargs="+", type=int)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument(
        "--max-schema-corrections",
        type=int,
        choices=(1, 2),
        default=1,
        help="explicitly raise to 2 only after inspecting a failed correction",
    )
    p.add_argument(
        "--repair-rejections",
        action="store_true",
        help="allow one audited cloud schema correction per rejected stage",
    )
    p = sub.add_parser("next-review")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--observation", action="append", default=[])
    p.add_argument(
        "--independent-identity",
        action="store_true",
        help="blind census plus ID association before the next visual review",
    )
    args = parser.parse_args()
    if args.command == "select":
        select(args.catalog, args.output)
        return
    root = args.root.resolve()
    rows = read(root / "selection.json")["examples"]
    if args.ids:
        rows = [row for row in rows if row["sequence"] in args.ids]
    if args.command == "next-review":
        for row in rows:
            next_review(root, row, args.observation, args.independent_identity)
        return
    results = []
    # Bounded scenes at once, with serial requests per scene: max 2 cloud calls.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(
                process, root, row, args.phase, args.repair_rejections, args.max_schema_corrections
            )
            for row in rows
        ]
        for future in as_completed(pending):
            results.append(future.result())
    save(root / f"batch_{args.phase}.json", results)


if __name__ == "__main__":
    main()
