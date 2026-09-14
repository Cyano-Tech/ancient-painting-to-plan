"""Evidence-typed ground landmarks and bounded cloud requests, CPU rendering only."""

import argparse
import base64
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from .advance_scene import load_prior
from .qwen_cloud import RUN_ROOT, SafeError, prepare_image, run
from .scene_schema import point
from .summarize_reviews import to_source_bbox

GROUND_FEATURES = {"wall_ground_corner", "column_ground_foot", "plinth_ground_corner"}
FEATURE_LEVELS = {
    **{f: "ground" for f in GROUND_FEATURES},
    "wall_occluder_contact": "occlusion",
    "upper_floor_corner": "upper",
    "eave_corner": "upper",
    "uncertain_feature": "unknown",
}


def validate_landmarks(data, context):
    if not isinstance(data.get("instances"), list) or not isinstance(data.get("rejected"), list):
        return ["instances and rejected must be lists"]
    expected = {o["id"] for o in context["candidates"]}
    accounted = set()
    all_points = set()
    instances = set()
    issues = []
    for inst in data["instances"]:
        oid = inst.get("id")
        if not isinstance(oid, str) or oid in instances:
            issues.append("Missing/duplicate instance ID")
        instances.add(str(oid))
        refs = inst.get("source_ids")
        if not isinstance(refs, list) or not refs:
            issues.append(f"{oid}: missing source IDs")
        else:
            for rid in refs:
                if rid not in expected or rid in accounted:
                    issues.append(f"{oid}: unknown/duplicate source ID")
                accounted.add(rid)
        if inst.get("identity_status") not in {"supported", "uncertain"} or inst.get(
            "ground_status"
        ) not in {"visible_partial", "occluded", "outside_crop", "uncertain"}:
            issues.append(f"{oid}: invalid identity/ground status")
        landmarks = inst.get("landmarks")
        segments = inst.get("segments")
        if not isinstance(landmarks, list) or not isinstance(segments, list):
            issues.append(f"{oid}: landmarks/segments must be lists")
            continue
        points = {}
        for lm in landmarks:
            lid = lm.get("id")
            if not isinstance(lid, str) or lid in all_points:
                issues.append(f"{oid}: missing/duplicate landmark ID")
            all_points.add(str(lid))
            points[lid] = lm
            if not point(lm.get("xy")):
                issues.append(f"{lid}: invalid point")
            if lm.get("feature") not in FEATURE_LEVELS or FEATURE_LEVELS.get(
                lm.get("feature")
            ) != lm.get("level"):
                issues.append(f"{lid}: feature/level mismatch")
        segment_ids = set()
        for seg in segments:
            sid = seg.get("id")
            a, b = points.get(seg.get("from")), points.get(seg.get("to"))
            if (
                not isinstance(sid, str)
                or sid in segment_ids
                or a is None
                or b is None
                or seg.get("from") == seg.get("to")
            ):
                issues.append(f"{oid}: invalid segment references")
                continue
            segment_ids.add(sid)
            role = seg.get("role")
            if role not in {"ground_edge", "occlusion_boundary", "upper_structure", "unknown"}:
                issues.append(f"{sid}: unknown role")
            if role == "ground_edge" and (a["level"] != "ground" or b["level"] != "ground"):
                issues.append(f"{sid}: non-ground endpoints promoted to ground")
    for rejected in data["rejected"]:
        rid = rejected.get("source_id")
        if rid not in expected or rid in accounted:
            issues.append("Unknown/duplicate rejected source")
        accounted.add(rid)
    if accounted != expected:
        issues.append("Input candidates were not fully accounted for")
    return issues


def validate_ground_stage(stage, data, context):
    try:
        if stage == "landmarks":
            return validate_landmarks(data, context)
        if stage == "ground_hypotheses":
            return validate_hypotheses(data, context)
        if stage == "ground_check":
            return validate_ground_check(data, context)
        return ["Unknown ground stage"]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return ["Malformed ground data: " + type(exc).__name__]


def normalize_landmark_roles(data):
    result = deepcopy(data)
    changes = []
    for inst in result.get("instances", []):
        for seg in inst.get("segments", []):
            if seg.get("role") == "visible_ground_line":
                seg["role"] = "ground_edge"
                changes.append(
                    {
                        "instance": inst["id"],
                        "segment": seg["id"],
                        "from": "visible_ground_line",
                        "to": "ground_edge",
                        "reason": "Explicit semantic alias only; coordinates and point levels unchanged.",
                    }
                )
    return result, changes


def save_landmark_normalization(run_dir):
    raw = json.loads((run_dir / "result.json").read_text())
    meta = json.loads((run_dir / "request_meta.json").read_text())
    if (
        meta["stage"] != "landmarks"
        or raw.get("finish_reason") != "stop"
        or not isinstance(raw.get("parsed"), dict)
    ):
        raise SafeError("Cannot normalize incomplete/non-landmark response")
    data, changes = normalize_landmark_roles(raw["parsed"])
    context = json.loads(Path(meta["context_source"]).read_text())
    issues = validate_landmarks(data, context)
    report = {
        "source_result": "result.json",
        "parsed": data,
        "normalization_changes": changes,
        "validation_issues": issues,
        "coordinates_changed": False,
        "point_levels_changed": False,
        "objects_added_or_removed": False,
    }
    (run_dir / "landmarks_normalized.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"changes": changes, "validation_issues": issues}, ensure_ascii=False))
    return 0 if not issues else 2


def convex_quad(points):
    if not isinstance(points, list) or len(points) != 4 or not all(point(p) for p in points):
        return False
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(c > 1e-5 for c in cross) or all(c < -1e-5 for c in cross)


def validate_hypotheses(data, context):
    prior = {o["id"]: o for o in context["instances"]}
    landmarks = {p["id"]: p for o in context["instances"] for p in o["landmarks"]}
    instances = data.get("instances")
    if not isinstance(instances, list):
        return ["instances must be a list"]
    issues = []
    seen = set()
    alternatives = set()
    for inst in instances:
        oid = inst.get("id")
        if oid not in prior or oid in seen:
            issues.append("Unknown/duplicate hypothesis instance")
            continue
        seen.add(oid)
        alts = inst.get("alternatives")
        if (
            not isinstance(alts, list)
            or len(alts) > 2
            or inst.get("status") not in {"proposed", "insufficient"}
        ):
            issues.append(f"{oid}: invalid proposal status/alternatives")
            continue
        if (inst["status"] == "proposed") != bool(alts):
            issues.append(f"{oid}: status disagrees with alternatives")
        own_landmarks = {p["id"] for p in prior[oid]["landmarks"]}
        for alt in alts:
            aid = alt.get("id")
            if not isinstance(aid, str) or (oid, aid) in alternatives:
                issues.append("Missing/duplicate alternative ID")
            alternatives.add((oid, str(aid)))
            ratio = alt.get("width_depth_ratio_range")
            if (
                not isinstance(ratio, list)
                or len(ratio) != 2
                or not all(type(x) in (int, float) and math.isfinite(x) for x in ratio)
                or not 0 < ratio[0] <= ratio[1]
            ):
                issues.append(f"{aid}: invalid ratio interval")
            for field, allowed in (
                ("reference_instance_ids", set(prior)),
                ("evidence_landmark_ids", set(landmarks)),
            ):
                if not isinstance(alt.get(field), list) or any(
                    r not in allowed for r in alt[field]
                ):
                    issues.append(f"{aid}: invalid {field}")
            corners = alt.get("image_corners")
            states = alt.get("vertex_status")
            refs = alt.get("vertex_landmark_ids")
            edges = alt.get("edge_status")
            if alt.get("placement") == "shape_only":
                if any(v != [] for v in (corners, states, refs, edges)):
                    issues.append(f"{aid}: shape-only has placed geometry")
                continue
            if alt.get("placement") not in {"anchored", "inferred"} or not convex_quad(corners):
                issues.append(f"{aid}: invalid placement or nonconvex quadrilateral")
                continue
            if not all(isinstance(v, list) and len(v) == 4 for v in (states, refs, edges)):
                issues.append(f"{aid}: missing four-corner provenance")
                continue
            if any(s not in {"visible", "inferred"} for s in states + edges):
                issues.append(f"{aid}: invalid edge/vertex status")
            for i, (state, rid) in enumerate(zip(states, refs)):
                if rid is not None and rid not in own_landmarks:
                    issues.append(f"{aid}: vertex references another or missing object")
                if state == "visible":
                    lm = landmarks.get(rid)
                    if not lm or lm["level"] != "ground" or lm["xy"] != corners[i]:
                        issues.append(f"{aid}: non-ground or displaced visible vertex")
                if edges[i] == "visible" and (
                    state != "visible" or states[(i + 1) % 4] != "visible"
                ):
                    issues.append(f"{aid}: visible edge has inferred endpoint")
            if alt["placement"] == "anchored" and "visible" not in states:
                issues.append(f"{aid}: anchored proposal lacks visible ground point")
    if seen != set(prior):
        issues.append("Hypotheses did not account for every instance")
    return issues


def validate_ground_check(data, context):
    expected = {
        (i["id"], a["id"]) for i in context["hypotheses"]["instances"] for a in i["alternatives"]
    }
    alternatives = {
        (i["id"], a["id"]): a for i in context["hypotheses"]["instances"] for a in i["alternatives"]
    }
    decisions = data.get("decisions")
    if not isinstance(decisions, list):
        return ["decisions must be a list"]
    seen = set()
    issues = []
    for d in decisions:
        pair = (d.get("instance_id"), d.get("alternative_id"))
        if (
            pair not in expected
            or pair in seen
            or d.get("verdict") not in {"usable_hypothesis", "needs_revision", "reject"}
        ):
            issues.append("Unknown/duplicate/invalid hypothesis decision")
        if context.get("require_corner_echo") and pair in alternatives:
            alt = alternatives[pair]
            if d.get("evaluated_corners") != alt["image_corners"]:
                issues.append("Checked corners do not match actual input order")
            if type(d.get("order_consistent")) is not bool:
                issues.append("Missing explicit order consistency")
            indices = d.get("front_edge_indices")
            if not isinstance(indices, list) or (
                indices
                and (
                    len(indices) != 2
                    or any(type(x) is not int or x not in range(4) for x in indices)
                    or indices[0] == indices[1]
                )
            ):
                issues.append("Invalid front-edge indices")
            if d.get("verdict") == "usable_hypothesis":
                wanted = [] if alt["placement"] == "shape_only" else [0, 1]
                if indices != wanted or d.get("order_consistent") is not True:
                    issues.append("Accepted hypothesis has inconsistent front edge")
        seen.add(pair)
    if seen != expected:
        issues.append("Hypothesis decisions are incomplete")
    return issues


def revalidate_ground_result(run_dir):
    raw_path = run_dir / "result.json"
    raw = json.loads(raw_path.read_text())
    meta = json.loads((run_dir / "request_meta.json").read_text())
    if (
        meta["stage"] not in {"landmarks", "ground_hypotheses", "ground_check"}
        or raw.get("finish_reason") != "stop"
        or not isinstance(raw.get("parsed"), dict)
    ):
        raise SafeError("Cannot revalidate incomplete ground response")
    context = json.loads(Path(meta["context_source"]).read_text())
    context = context.get("parsed", context)
    if meta["stage"] == "ground_check":
        parent = Path(meta["context_source"]).parent
        parent_meta = json.loads((parent / "request_meta.json").read_text())
        landmark_data = json.loads(Path(parent_meta["context_source"]).read_text())
        landmarks = landmark_data.get("parsed", landmark_data)
        context = {"landmarks": landmarks, "hypotheses": context, **meta.get("context_flags", {})}
    issues = validate_ground_stage(meta["stage"], raw["parsed"], context)
    report = {
        "source_result": "result.json",
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "parsed": raw["parsed"],
        "validation_issues": issues,
        "original_validation_issues": raw.get("validation_issues", []),
        "validation_note": "Alternative identity is (instance_id, alternative_id); IDs need only be unique within each instance.",
        "coordinates_changed": False,
        "objects_added_or_removed": False,
        "raw_answer_changed": False,
    }
    (run_dir / (meta["stage"] + "_validated.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"run": str(run_dir), "validation_issues": issues}, ensure_ascii=False))
    return 0 if not issues else 2


def rasterize(svg, output):
    """Render our internally generated SVG on CPU, without desktop services.

    CairoSVG is an optional dependency. Keep unsafe XML processing disabled;
    this helper is for our generated diagrams, not arbitrary remote SVG input.
    """
    try:
        from cairosvg import svg2png
    except ImportError as exc:
        raise RuntimeError("PNG output needs the 'render' extra: pip install '.[render]'") from exc
    svg2png(bytestring=svg.encode("utf-8"), write_to=str(output))


def coordinate_guide(encoded, size, out):
    w, h = size
    uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><image width="{w}" height="{h}" href="{uri}"/>'
    ]
    font = max(12, round(min(w, h) * 0.026))
    for k in range(100, 1000, 100):
        x, y = k * w / 1000, k * h / 1000
        svg.append(
            f'<path d="M{x},0 V{h} M0,{y} H{w}" stroke="#70f5ef" stroke-opacity=".4" stroke-width=".8"/>'
        )
        for tx, ty, label in ((x + 2, font + 3, f"x{k}"), (2, y - 2, f"y{k}")):
            svg.append(
                f'<text x="{tx}" y="{ty}" font-family="sans-serif" font-size="{font}" fill="#d0fff4" stroke="#162e29" stroke-width="2.5" paint-order="stroke">{label}</text>'
            )
    svg.append("</svg>")
    text = "".join(svg)
    out.with_suffix(".svg").write_text(text)
    rasterize(text, out)


def expanded_crop(meta, horizontal=0.12, top=0.15, bottom=0.45):
    a, b, c, d = meta["crop_in_source_pixels"]
    w, h = meta["source_size_after_exif"]
    return [
        max(0, round(a - (c - a) * horizontal)),
        max(0, round(b - (d - b) * top)),
        min(w, round(c + (c - a) * horizontal)),
        min(h, round(d + (d - b) * bottom)),
    ]


def remap_bbox(box, old_crop, new_crop):
    a, b, c, d = to_source_bbox(box, old_crop)
    x0, y0, x1, y1 = new_crop
    return [
        (a - x0) * 1000 / (x1 - x0),
        (b - y0) * 1000 / (y1 - y0),
        (c - x0) * 1000 / (x1 - x0),
        (d - y0) * 1000 / (y1 - y0),
    ]


def landmarks_job(prior_dir, label, work):
    meta, data, _ = load_prior(prior_dir)
    im = meta["image"]
    if hashlib.sha256(Path(im["source_path"]).read_bytes()).hexdigest() != im["source_sha256"]:
        raise SafeError("Source changed since prior run")
    crop = expanded_crop(im)
    context = {
        "stage": "candidate_locations_only",
        "coordinate_system": "primary_image_normalized_0_1000",
        "candidates": [
            {
                "id": o["id"],
                "location_hint_bbox": remap_bbox(o["bbox"], im["crop_in_source_pixels"], crop),
            }
            for o in data["objects"]
            if o["category"] == "house"
        ],
    }
    directory = work / label
    directory.mkdir()
    context_path = directory / "context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    encoded, image_meta = prepare_image(im["source_path"], 2400, crop)
    guide = directory / "coordinate_guide.png"
    coordinate_guide(encoded, image_meta["transmitted_size"], guide)
    args = argparse.Namespace(
        stage="landmarks",
        image=im["source_path"],
        crop=crop,
        base_url=None,
        run_label=label,
        context=context,
        context_source=str(context_path),
        reference_images=[
            {
                "image": str(guide),
                "label": "同图1，同尺寸同范围；只是0–1000图像坐标刻度，不是地面网格。",
            }
        ],
    )
    code = run(args)
    result = json.loads((Path(args.output_dir) / "result.json").read_text())
    return {
        "label": label,
        "prior_dir": str(prior_dir),
        "output_dir": args.output_dir,
        "exit_code": code,
        "crop_source_pixels": crop,
        "usage": result["usage"],
        "validation_issues": result["validation_issues"],
    }


def render_landmarks(run_dir):
    run_dir = Path(run_dir).resolve()
    meta, data, _ = load_prior(run_dir)
    if meta["stage"] != "landmarks":
        raise ValueError("Expected landmarks run")
    iw, ih = meta["image"]["transmitted_size"]
    w = 1200
    h = round(w * ih / iw)
    colors = {"ground": "#7aff8a", "occlusion": "#ffb15c", "upper": "#75dfff", "unknown": "#e6d1ef"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="{h + 145}"><rect width="1240" height="{h + 145}" fill="#182c24"/>',
        '<text x="20" y="35" fill="#eff8e3" font-family="Noto Sans CJK SC,sans-serif" font-size="24">接地逐点核对 · 仍为模型判断</text>',
        '<text x="20" y="67" fill="#c1d4c9" font-family="Noto Sans CJK SC,sans-serif" font-size="17">绿：模型认为接地　橙：遮挡交界　蓝：楼层/檐口　紫：不确定；都需对照原画</text>',
        f'<image x="20" y="90" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode((run_dir / "input.jpg").read_bytes()).decode()}"/>',
    ]
    for inst in data["instances"]:
        by_id = {p["id"]: p for p in inst["landmarks"]}
        for s in inst["segments"]:
            a, b = by_id[s["from"]]["xy"], by_id[s["to"]]["xy"]
            color = "#7aff8a" if s["role"] == "ground_edge" else "#ffb15c"
            svg.append(
                f'<line x1="{20 + a[0] * w / 1000}" y1="{90 + a[1] * h / 1000}" x2="{20 + b[0] * w / 1000}" y2="{90 + b[1] * h / 1000}" stroke="{color}" stroke-width="2.3"/>'
            )
        for p in inst["landmarks"]:
            x, y = 20 + p["xy"][0] * w / 1000, 90 + p["xy"][1] * h / 1000
            label = inst["id"] + ":" + p["id"]
            svg.append(
                f'<g><title>{html.escape(p["evidence"])}</title><circle cx="{x}" cy="{y}" r="5" fill="#183127" stroke="{colors[p["level"]]}" stroke-width="2"/><text x="{x + 7}" y="{y - 7}" fill="{colors[p["level"]]}" stroke="#163329" stroke-width="3" paint-order="stroke" font-family="sans-serif" font-size="14">{label}</text></g>'
            )
    svg.append(
        f'<text x="20" y="{h + 126}" fill="#d8c9ae" font-family="Noto Sans CJK SC,sans-serif" font-size="17">接地点与遮挡交界分开保存；本图不是地基多边形，也不是俯视图。</text></svg>'
    )
    source = "".join(svg)
    (run_dir / "landmarks.svg").write_text(source)
    rasterize(source, run_dir / "landmarks.png")
    print(run_dir / "landmarks.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reviews", type=Path, help="Manifest of independent detail reviews")
    inputs.add_argument("--render-landmarks", type=Path)
    inputs.add_argument("--normalize-landmarks", type=Path)
    inputs.add_argument("--revalidate", type=Path)
    args = p.parse_args()
    if args.revalidate:
        return revalidate_ground_result(args.revalidate)
    if args.normalize_landmarks:
        return save_landmark_normalization(args.normalize_landmarks)
    if args.render_landmarks:
        render_landmarks(args.render_landmarks)
        return 0
    jobs = json.loads(args.reviews.read_text())["tiles"]
    for job in jobs:
        load_prior(job["output_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = RUN_ROOT / (stamp + "_landmarks_batch")
    out.mkdir(mode=0o700)
    manifest = {
        "stage": "landmarks",
        "tiles": [],
        "automatic_retries": 0,
        "local_model_used": False,
        "gpu_used": False,
    }
    print("Landmarks batch:", out, flush=True)

    def execute(job):
        try:
            return landmarks_job(job["output_dir"], job["label"], out)
        except Exception as exc:
            return {"label": job["label"], "exit_code": 1, "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(execute, jobs):
            manifest["tiles"].append(result)
            (out / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(r["exit_code"] == 0 for r in manifest["tiles"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
