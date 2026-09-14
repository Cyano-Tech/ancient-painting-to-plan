"""Cloud-only full-scene analysis and reproducible whole-plan refinement.

No image-specific counts, coordinates or corrections live in this code.
Inference artifacts, including geometry, remain separate versioned JSON files.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps
from .inventory_tiles import tile_boxes
from .plan_schema import validate_plan_stage, validate_terrain
from .qwen_cloud import RUN_ROOT, SafeError, run, model_matches


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stamp(label):
    folder = RUN_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + label)
    folder.mkdir(parents=True, mode=0o700)
    return folder


def accepted(folder):
    result = read(Path(folder) / "result.json")
    if (
        result.get("validation_issues")
        or result.get("finish_reason") != "stop"
        or not model_matches("qwen3.8-max", result.get("model_returned"))
    ):
        raise SafeError(f"Unusable cloud result: {folder}")
    return result["parsed"]


def request(stage, image, crop=None, context=None, label=None, references=None, folder=None):
    context_path = None
    if context is not None:
        folder = folder or stamp(stage + "_context")
        context_path = folder / "context.json"
        save(context_path, context)
    args = argparse.Namespace(
        stage=stage,
        image=str(image),
        crop=crop,
        base_url=None,
        run_label=label,
        context=context,
        context_source=str(context_path) if context_path else None,
        reference_images=references or [],
    )
    code = run(args)
    return {"stage": stage, "crop": crop, "label": label, "exit_code": code, "run": args.output_dir}


def start(image):
    image = Path(image).resolve()
    with Image.open(image) as im:
        size = ImageOps.exif_transpose(im).size
    batch = stamp("complete_plan_survey")
    manifest = {
        "source": {
            "path": str(image),
            "size": list(size),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
        "cloud_model": "qwen3.8-max",
        "gpu_used": False,
        "jobs": [],
        "batch": str(batch),
    }
    jobs = [("plan_terrain", None, "whole")] + [
        ("plan_buildings", c, n) for n, c in tile_boxes(*size)
    ]

    def job(spec):
        stage, crop, label = spec
        try:
            return request(stage, image, crop, label=label)
        except (SafeError, OSError, ValueError) as exc:
            return {
                "stage": stage,
                "crop": crop,
                "label": label,
                "exit_code": 1,
                "error_type": type(exc).__name__,
            }

    save(batch / "manifest.json", manifest)
    print("Survey manifest:", batch / "manifest.json", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(job, jobs):
            manifest["jobs"].append(result)
            save(batch / "manifest.json", manifest)
    print(batch / "manifest.json", flush=True)
    return batch


def map_point(p, crop, size):
    return [
        round((crop[0] + p[0] * (crop[2] - crop[0]) / 1000) / size[0] * 1000, 3),
        round((crop[1] + p[1] * (crop[3] - crop[1]) / 1000) / size[1] * 1000, 3),
    ]


def layout(manifest):
    m = read(manifest)
    source = m["source"]
    if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
        raise SafeError("Source has changed")
    context = {"source": source, "candidates": [], "terrain": None, "tile_coverage": []}
    for job in m["jobs"]:
        if job["exit_code"]:
            raise SafeError("Incomplete survey; inspect failed jobs before layout")
        data = accepted(job["run"])
        if job["stage"] == "plan_terrain":
            context["terrain"] = data
            continue
        context["tile_coverage"].append(
            {"tile": job["label"], "notes": data["coverage_notes"], "exclusions": data["excluded"]}
        )
        for raw in data["objects"]:
            b = deepcopy(raw)
            b["id"] = job["label"] + ":" + b["id"]
            b["source_run"] = job["run"]
            b["bbox"] = map_point(raw["bbox"][:2], job["crop"], source["size"]) + map_point(
                raw["bbox"][2:], job["crop"], source["size"]
            )
            b["anchor"] = map_point(raw["anchor"], job["crop"], source["size"])
            b["footprint_image"] = [
                map_point(p, job["crop"], source["size"]) for p in raw["footprint_image"]
            ]
            context["candidates"].append(b)
    if context["terrain"] is None:
        raise SafeError("Missing terrain stage")
    result = request("plan_layout", source["path"], context=context, label="whole")
    if result["exit_code"]:
        return result
    folder = stamp("whole_plan_v1")
    plan = {
        "schema_version": 1,
        "source": source,
        "terrain": context["terrain"],
        "layout": accepted(result["run"]),
        "survey_manifest": str(Path(manifest).resolve()),
        "candidate_context": str(Path(result["run"]) / "request_meta.json"),
        "layout_run": result["run"],
        "revisions": [],
        "camera_calibration_verified": False,
        "metric_units": False,
        "gpu_used": False,
    }
    save(folder / "plan.json", plan)
    print("Whole plan:", folder / "plan.json", flush=True)
    return folder


def critique(plan_path, details=None, notes=None):
    from .render_complete_plan import geometry_checks

    p = Path(plan_path).resolve()
    plan = read(p)
    for name in ("topview.png", "source_overlay.png"):
        if not (p.parent / name).exists():
            raise SafeError("Render plan before visual review")
    context = {
        "terrain": plan["terrain"],
        "layout": plan["layout"],
        "geometry_checks": geometry_checks(plan),
        "plan_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "plan_path": str(p),
        "draft_validation_issues": plan.get("draft_validation_issues", []),
        "previous_reviews": [
            {"verdict": r.get("verdict"), "issues": r.get("issues")} for r in plan["revisions"]
        ],
    }
    if details:
        detail_data = read(details)
        if detail_data["source"] != plan["source"] or any(
            j["exit_code"] for j in detail_data["jobs"]
        ):
            raise SafeError("Incomplete or unrelated detail evidence")
        context["detail_evidence"] = detail_data["evidence"]
        candidates = [
            {"id": d["id"], "bbox": d["bbox"]}
            for e in detail_data["evidence"]
            for d in e["decisions"]
            if d["status"] in {"keep", "uncertain"}
        ]
        for e in detail_data["evidence"]:
            candidates += [
                {
                    "id": Path(e["run"]).name + ":" + d["id"],
                    "bbox": d["bbox"],
                    "new_candidate": True,
                }
                for d in e["missing"]
            ]
        overlap_hints = []
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                x, y, u, v = a["bbox"]
                xx, yy, uu, vv = b["bbox"]
                fraction = (
                    max(0, min(u, uu) - max(x, xx))
                    * max(0, min(v, vv) - max(y, yy))
                    / min((u - x) * (v - y), (uu - xx) * (vv - yy))
                )
                if fraction > 0.4:
                    overlap_hints.append(
                        {
                            "ids": [a["id"], b["id"]],
                            "smaller_bbox_overlap": round(fraction, 3),
                            "note": "Only a review trigger, not automatic identity merge.",
                        }
                    )
        context["detail_overlap_hints"] = overlap_hints
    refs = [
        {
            "image": str(p.parent / "topview.png"),
            "label": "候选统一俯视总图，同一共享坐标。箭头为正面，虚线为推测。",
        },
        {
            "image": str(p.parent / "source_overlay.png"),
            "label": "原画对应的ID/地基假设，线条不是事实，必须用第1张原画核对。",
        },
    ]
    if notes:
        notes_data = read(notes)
        if notes_data["source_sha256"] != plan["source"]["sha256"]:
            raise SafeError("Review notes source mismatch")
        context["agent_visual_review"] = notes_data
        if notes_data.get("previous_unaccepted_review"):
            failed = read(Path(notes_data["previous_unaccepted_review"]) / "result.json")
            if failed.get("finish_reason") != "stop" or not isinstance(failed.get("parsed"), dict):
                raise SafeError("Cannot use truncated failed review")
            context["previous_unaccepted_review"] = {
                "parsed": failed["parsed"],
                "validation_issues": failed["validation_issues"],
            }
        focus = notes_data.get("focus_ids", [])
        if notes_data.get("focus_reference"):
            refs.append(
                {
                    "image": notes_data["focus_reference"],
                    "label": "原画上下文放大用于核实疑似倒影/漏检，不是重绘图；需独立检查。",
                }
            )
        elif focus:
            from .render_complete_plan import render_focus

            focus_path = render_focus(plan, focus, p.parent / "review_focus.png")
            refs.append(
                {
                    "image": str(focus_path),
                    "label": "仅对待核实对象共同区域放大，彩框来自旧候选并非事实。请看结构连续性，不顺着框数房屋。",
                }
            )
    result = request(
        "plan_critique", plan["source"]["path"], context=context, label="whole", references=refs
    )
    save(p.parent / "latest_critique.json", result)
    print("Critique:", result, flush=True)
    return result


def apply_review(plan_path, review_run, repair_accounting=False):
    old = read(plan_path)
    check = accepted(review_run)
    p = deepcopy(old)
    meta = read(Path(review_run) / "request_meta.json")
    review_context = read(meta["context_source"])
    if (
        review_context.get("plan_sha256")
        != hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    ):
        raise SafeError("Review belongs to a different plan version")
    context = {"terrain": old["terrain"], "layout": old["layout"]}
    issues = validate_plan_stage("plan_critique", check, context)
    if issues:
        raise SafeError("; ".join(issues))
    bmap = {b["id"]: b for b in p["layout"]["buildings"]}
    if check.get("replace_rejected") is not None:
        p["layout"]["rejected"] = check["replace_rejected"]
    if check.get("replace_relations") is not None:
        p["layout"]["relations"] = check["replace_relations"]
    for b in check["replace_buildings"] + check["add_buildings"]:
        bmap[b["id"]] = b
    for rem in check["remove_buildings"]:
        obj = bmap.pop(rem["id"])
        if rem.get("merge_into"):
            dest = bmap[rem["merge_into"]]
            dest["source_ids"] = sorted(set(dest["source_ids"] + obj["source_ids"]))
        else:
            p["layout"]["rejected"] += [
                {"source_id": sid, "reason": rem["reason"]} for sid in obj["source_ids"]
            ]
    p["layout"]["buildings"] = list(bmap.values())
    from .apply_identity_evidence import preserve_observation_locks

    preserve_observation_locks(old, p)
    # Redirect relations for explicitly merged identities; drop only deleted endpoints.
    merged = {r["id"]: r.get("merge_into") for r in check["remove_buildings"]}
    relations = []
    for rel in p["layout"]["relations"]:
        rel = {
            **rel,
            "from": merged.get(rel["from"], rel["from"]),
            "to": merged.get(rel["to"], rel["to"]),
        }
        if rel["from"] and rel["to"] and rel["from"] != rel["to"]:
            relations.append(rel)
    p["layout"]["relations"] = relations
    for key in ("terrain", "zones", "trees", "routes"):
        values = {v["id"]: v for v in p["terrain"][key]}
        for val in check["replace_" + key]:
            values[val["id"]] = val
        for rem in check.get("remove_elements", []):
            if rem["collection"] == key:
                if rem["id"] not in values or not rem.get("reason"):
                    raise SafeError("Invalid terrain removal")
                del values[rem["id"]]
        p["terrain"][key] = list(values.values())
    expected = {sid for b in old["layout"]["buildings"] for sid in b["source_ids"]} | {
        r["source_id"] for r in old["layout"]["rejected"]
    }
    issues = validate_terrain(p["terrain"]) + validate_plan_stage(
        "plan_layout",
        p["layout"],
        {"candidates": [{"id": s} for s in expected], "terrain": p["terrain"]},
    )
    if (
        issues == ["Every source candidate must be accounted for exactly once"]
        and repair_accounting
    ):
        context = {
            "expected_source_ids": sorted(expected),
            "layout": p["layout"],
            "validation_issues": issues,
        }
        record = request("plan_accounting", p["source"]["path"], context=context, label="repair")
        assignment = accepted(record["run"])
        for b in p["layout"]["buildings"]:
            b["source_ids"] = []
        p["layout"]["rejected"] = []
        for a in assignment["assignments"]:
            if a["target"] is None:
                p["layout"]["rejected"].append({"source_id": a["source_id"], "reason": a["reason"]})
            else:
                bmap[a["target"]]["source_ids"].append(a["source_id"])
        p["accounting_repairs"] = p.get("accounting_repairs", []) + [record]
        issues = validate_plan_stage(
            "plan_layout",
            p["layout"],
            {"candidates": [{"id": s} for s in expected], "terrain": p["terrain"]},
        )
    if issues:
        raise SafeError("Refusing invalid revision: " + "; ".join(issues))
    p.pop("draft_validation_issues", None)
    p["revisions"].append(
        {
            "run": str(Path(review_run).resolve()),
            "verdict": check["verdict"],
            "issues": check["issues"],
            "checks": check["checks"],
            "remaining_uncertainties": check["remaining_uncertainties"],
        }
    )
    p["parent_plan"] = str(Path(plan_path).resolve())
    folder = stamp("whole_plan_v" + str(len(p["revisions"]) + 1))
    save(folder / "plan.json", p)
    print("Revised plan:", folder / "plan.json", flush=True)
    return folder


def draft(layout_run, manifest):
    """Retain invalid draft explicitly for correction; never mark it accepted."""
    raw = read(Path(layout_run) / "result.json")
    meta = read(Path(layout_run) / "request_meta.json")
    if (
        raw.get("finish_reason") != "stop"
        or not model_matches("qwen3.8-max", raw.get("model_returned"))
        or not isinstance(raw.get("parsed"), dict)
    ):
        raise SafeError("Cannot repair truncated/non-JSON/wrong-model output")
    context = read(meta["context_source"])
    m = read(manifest)
    if context["source"] != m["source"]:
        raise SafeError("Survey/source mismatch")
    p = {
        "schema_version": 1,
        "source": m["source"],
        "terrain": context["terrain"],
        "layout": raw["parsed"],
        "survey_manifest": str(Path(manifest).resolve()),
        "layout_run": str(Path(layout_run).resolve()),
        "draft_validation_issues": raw["validation_issues"],
        "revisions": [],
        "camera_calibration_verified": False,
        "metric_units": False,
        "gpu_used": False,
    }
    folder = stamp("whole_plan_invalid_draft")
    save(folder / "plan.json", p)
    print(folder / "plan.json", flush=True)
    return folder


def details(plan_path):
    plan = read(plan_path)
    source = plan["source"]
    sw, sh = source["size"]
    groups = {}
    # Generic 3x3 ownership partition; actual request crop encloses each group's
    # full candidates with context, so no object is clipped at partition seams.
    for b in plan["layout"]["buildings"]:
        x, y = b["source_anchor"]
        key = (min(2, int(x * 3 / 1000)), min(2, int(y * 3 / 1000)))
        groups.setdefault(key, []).append(b)
    folder = stamp("whole_plan_detail_survey")
    manifest = {
        "source": source,
        "plan_path": str(Path(plan_path).resolve()),
        "jobs": [],
        "evidence": [],
    }

    def job(entry):
        key, buildings = entry
        bounds = [
            min(b["source_bbox"][0] for b in buildings),
            min(b["source_bbox"][1] for b in buildings),
            max(b["source_bbox"][2] for b in buildings),
            max(b["source_bbox"][3] for b in buildings),
        ]
        px = max(20, (bounds[2] - bounds[0]) * 0.1)
        py = max(20, (bounds[3] - bounds[1]) * 0.15)
        crop = [
            max(0, int((bounds[0] - px) * sw / 1000)),
            max(0, int((bounds[1] - py) * sh / 1000)),
            min(sw, int((bounds[2] + px) * sw / 1000)),
            min(sh, int((bounds[3] + py) * sh / 1000)),
        ]

        def local(p):
            return [
                max(
                    0,
                    min(1000, round((p[0] * sw / 1000 - crop[0]) / (crop[2] - crop[0]) * 1000, 3)),
                ),
                max(
                    0,
                    min(1000, round((p[1] * sh / 1000 - crop[1]) / (crop[3] - crop[1]) * 1000, 3)),
                ),
            ]

        context = {
            "candidates": [
                {
                    "id": b["id"],
                    "bbox": local(b["source_bbox"][:2]) + local(b["source_bbox"][2:]),
                    "anchor": local(b["source_anchor"]),
                    "footprint_image": [local(p) for p in b["source_footprint"]],
                }
                for b in buildings
            ]
        }
        result = request(
            "plan_details", source["path"], crop, context, label=f"cell_{key[0]}_{key[1]}"
        )
        result["candidate_ids"] = [b["id"] for b in buildings]
        return result

    print("Detail manifest:", folder / "manifest.json", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(job, groups.items()):
            manifest["jobs"].append(result)
            if result["exit_code"] == 0:
                data = accepted(result["run"])
                for b in data["decisions"] + data["missing"]:
                    b["bbox"] = map_point(
                        b["bbox"][:2], result["crop"], source["size"]
                    ) + map_point(b["bbox"][2:], result["crop"], source["size"])
                    b["anchor"] = map_point(b["anchor"], result["crop"], source["size"])
                    b["footprint_image"] = [
                        map_point(p, result["crop"], source["size"]) for p in b["footprint_image"]
                    ]
                manifest["evidence"].append(
                    {"run": result["run"], "coordinate_system": "full_source_1000", **data}
                )
            save(folder / "manifest.json", manifest)
    print(folder / "manifest.json", flush=True)
    return folder


def focus_details(plan_path, ids, questions=None):
    from .render_complete_plan import render_focus

    plan = read(plan_path)
    source = plan["source"]
    sw, sh = source["size"]
    objects = [b for b in plan["layout"]["buildings"] if b["id"] in ids]
    if {b["id"] for b in objects} != set(ids):
        raise SafeError("Unknown focus IDs")
    folder = stamp("identity_focus_details")
    x0 = max(0, min(b["source_bbox"][0] for b in objects) - 25)
    y0 = max(0, min(b["source_bbox"][1] for b in objects) - 25)
    x1 = min(1000, max(b["source_bbox"][2] for b in objects) + 25)
    y1 = min(1000, max(b["source_bbox"][3] for b in objects) + 25)
    crop = [
        int(x0 * sw / 1000),
        int(y0 * sh / 1000),
        min(sw, math.ceil(x1 * sw / 1000)),
        min(sh, math.ceil(y1 * sh / 1000)),
    ]
    if questions:
        from .identity_audit import audit_crop

        crop = audit_crop(objects, source["size"])

    def local(p):
        return [
            round((p[0] * sw / 1000 - crop[0]) * 1000 / (crop[2] - crop[0]), 3),
            round((p[1] * sh / 1000 - crop[1]) * 1000 / (crop[3] - crop[1]), 3),
        ]

    context = {
        "candidates": [
            {
                "id": b["id"],
                "bbox": local(b["source_bbox"][:2]) + local(b["source_bbox"][2:]),
                "anchor": local(b["source_anchor"]),
                "footprint_image": [local(p) for p in b["source_footprint"]],
            }
            for b in objects
        ],
        "instruction_note": "编号只绑定彩框，不绑定旧名称/旧证据。每个框究竟圈到哪片屋顶和哪段墙身必须对上；不能看见别处人物便归给本框。",
    }
    if questions:
        q = read(questions)
        if q.get("source_sha256") != source["sha256"]:
            raise SafeError("Unrelated image questions")
        context.update(identity_audit=True, observation_questions=q)
        # Do not prime an independent appearance review with former foundations.
        context["candidates"] = [
            {"id": b["id"], "bbox": local(b["source_bbox"][:2]) + local(b["source_bbox"][2:])}
            for b in objects
        ]
    focus = render_focus(plan, ids, folder / "focus.png", crop=crop)
    job = request(
        "plan_details",
        source["path"],
        crop,
        context,
        label="identity_focus",
        references=[
            {
                "image": str(focus),
                "label": "同一局部的ID彩框位置参考，没有旧类别或旧描述；框也可能不准。",
            }
        ],
    )
    data = accepted(job["run"])
    for b in data["decisions"] + data["missing"]:
        b["bbox"] = map_point(b["bbox"][:2], crop, source["size"]) + map_point(
            b["bbox"][2:], crop, source["size"]
        )
        b["anchor"] = map_point(b["anchor"], crop, source["size"])
        b["footprint_image"] = [map_point(p, crop, source["size"]) for p in b["footprint_image"]]
    manifest = {
        "source": source,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "jobs": [job],
        "evidence": [{"run": job["run"], "coordinate_system": "full_source_1000", **data}],
    }
    save(folder / "manifest.json", manifest)
    print(folder / "manifest.json", flush=True)
    return folder


def final_check(plan_path):
    from .render_complete_plan import geometry_checks, rect_corners

    p = Path(plan_path).resolve()
    plan = read(p)
    if plan.get("draft_validation_issues"):
        raise SafeError("Invalid draft cannot enter final review")
    # Historical prose contains superseded counts, collisions and shapes. The
    # final reviewer sees only current observations and computed geometry.
    keys = (
        "id",
        "source_ids",
        "category",
        "subtype",
        "label",
        "source_bbox",
        "source_anchor",
        "source_footprint",
        "zone_id",
        "plan_center",
        "plan_size",
        "front_clock",
        "ratio_range",
        "confidence",
        "evidence",
        "assumption",
        "grouped",
        "unit_count_range",
    )
    current_layout = {
        **plan["layout"],
        "buildings": [{k: b[k] for k in keys if k in b} for b in plan["layout"]["buildings"]],
    }
    context = {
        "plan_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "plan_path": str(p),
        "terrain": plan["terrain"],
        "layout": current_layout,
        "geometry_checks": geometry_checks(plan),
        "geometric_facts": {
            "active_footprint_regions": len(plan["layout"]["buildings"]),
            "pending_candidates": len(plan.get("pending_candidates", [])),
            "grouped_ids": [b["id"] for b in plan["layout"]["buildings"] if b.get("grouped")],
            "rect_corners": {b["id"]: rect_corners(b) for b in plan["layout"]["buildings"]},
        },
    }
    context["pending_candidates"] = plan.get("pending_candidates", [])
    context["identity_audit_summary"] = [
        {
            "id": c["id"],
            "action": c["action"],
            "evidence": c.get("evidence", c.get("after_observation", {}).get("evidence")),
        }
        for a in plan.get("identity_audits", [])
        for c in a["changes"]
    ]
    context["review_scope_note"] = (
        "未定候选只显示紫色问号位置，不生成地基、不计栋数。已排除倒影没有另一个实体地基；水岸边界精度仍有限，不能声称已精确恢复。不能因旧全图审查一度通过便忽略本轮局部反证。"
    )
    refs = [
        {
            "image": str(p.parent / "topview.png"),
            "label": "实际渲染的完整共享俯视图，不是独立房屋形状清单",
        },
        {
            "image": str(p.parent / "source_overlay.png"),
            "label": "原画对应ID与推测地基，虚线不是已证实边界",
        },
    ]
    result = request(
        "plan_final_check", plan["source"]["path"], context=context, label="whole", references=refs
    )
    save(p.parent / "final_check_reference.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("survey").add_argument("--image", required=True, type=Path)
    sub.add_parser("layout").add_argument("--manifest", required=True, type=Path)
    a = sub.add_parser("critique")
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--details", type=Path)
    a.add_argument("--notes", type=Path)
    sub.add_parser("details").add_argument("--plan", required=True, type=Path)
    a = sub.add_parser("focus")
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--ids", nargs="+", required=True)
    a.add_argument("--questions", type=Path)
    sub.add_parser("final-check").add_argument("--plan", required=True, type=Path)
    a = sub.add_parser("apply")
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--review", required=True, type=Path)
    a.add_argument("--repair-accounting", action="store_true")
    a = sub.add_parser("draft")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "survey":
        start(args.image)
    elif args.command == "layout":
        layout(args.manifest)
    elif args.command == "critique":
        critique(args.plan, args.details, args.notes)
    elif args.command == "details":
        details(args.plan)
    elif args.command == "focus":
        focus_details(args.plan, args.ids, args.questions)
    elif args.command == "final-check":
        final_check(args.plan)
    elif args.command == "draft":
        draft(args.run, args.manifest)
    else:
        apply_review(args.plan, args.review, args.repair_accounting)
