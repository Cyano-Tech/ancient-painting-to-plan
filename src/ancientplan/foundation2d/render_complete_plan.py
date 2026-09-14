"""CPU-only SVG/HTML plan rendering and geometry diagnostics (not accuracy scores)."""

import argparse
import base64
from collections import Counter
import html
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .grounding import rasterize
from .qwen_cloud import prepare_image

COLORS = {"mountain": "#a7b4a0", "water": "#a9cbd3", "flat": "#e5ddbf", "other": "#c3beb0"}


def esc(v):
    return html.escape(str(v), quote=True)


def pts(poly):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in poly)


def text(x, y, t, size=13, fill="#233e35", extra=""):
    """Use two text passes for outlines, including SVG 1.1-only renderers.

    CairoSVG does not implement SVG 2 paint-order; a thick stroke drawn after
    the fill can hide small house IDs. An explicit halo followed by the fill
    has the same appearance in both browser SVG and CPU PNG exports.
    """
    attributes = f'x="{x}" y="{y}" font-size="{size}"'
    if 'paint-order="stroke"' in extra:
        halo = f'<text {attributes} fill="none" {extra}>{esc(t)}</text>'
        foreground = re.sub(r'\b(?:paint-order|stroke|stroke-width)="[^"]*"', "", extra)
        return halo + f'<text {attributes} fill="{fill}" {foreground}>{esc(t)}</text>'
    return f'<text {attributes} fill="{fill}" {extra}>{esc(t)}</text>'


def rect_corners(b):
    angle = math.pi * b["front_clock"] / 6
    f = (math.sin(angle), -math.cos(angle))
    r = (-math.cos(angle), -math.sin(angle))
    x, y = b["plan_center"]
    w, d = b["plan_size"]
    return [
        [x + sr * w / 2 * r[0] + sf * d / 2 * f[0], y + sr * w / 2 * r[1] + sf * d / 2 * f[1]]
        for sr, sf in ((-1, 1), (1, 1), (1, -1), (-1, -1))
    ]


def contains(p, poly):
    x, y = p
    inside = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def overlap(a, b):
    """Convex separating-axis test, positive-area contact only."""
    for poly in (a, b):
        for p, q in zip(poly, poly[1:] + poly[:1]):
            axis = (-(q[1] - p[1]), q[0] - p[0])
            av = [sum(v * t for v, t in zip(x, axis)) for x in a]
            bv = [sum(v * t for v, t in zip(x, axis)) for x in b]
            if max(av) <= min(bv) + 1e-6 or max(bv) <= min(av) + 1e-6:
                return False
    return True


def geometry_checks(plan):
    buildings = plan["layout"]["buildings"]
    terrain = plan["terrain"]
    zones = {z["id"]: z for z in terrain["zones"]}
    checks = []
    for i, b in enumerate(buildings):
        corners = rect_corners(b)
        for c in buildings[i + 1 :]:
            if overlap(corners, rect_corners(c)):
                checks.append(
                    {
                        "kind": "building_overlap",
                        "ids": [b["id"], c["id"]],
                        "severity": "major",
                        "note": "Footprints overlap in shared 2D plan; review identity, placement or distinct support levels.",
                    }
                )
        zone = zones[b["zone_id"]]
        if not contains(b["plan_center"], zone["plan_polygon"]):
            checks.append(
                {
                    "kind": "center_outside_support_zone",
                    "ids": [b["id"], zone["id"]],
                    "severity": "major",
                }
            )
        outside = sum(not contains(p, zone["plan_polygon"]) for p in corners)
        if outside:
            checks.append(
                {
                    "kind": "corners_outside_support_zone",
                    "ids": [b["id"], zone["id"]],
                    "corners": outside,
                    "severity": "minor",
                }
            )
        if any(not contains(p, terrain["plan_boundary"]) for p in corners):
            checks.append({"kind": "outside_plan_boundary", "ids": [b["id"]], "severity": "major"})
        for t in terrain["terrain"]:
            if t["category"] == "water" and contains(b["plan_center"], t["plan_polygon"]):
                checks.append(
                    {
                        "kind": "center_in_water",
                        "ids": [b["id"], t["id"]],
                        "severity": "major",
                        "note": "May be water pavilion or bridge; evidence required.",
                    }
                )
    return checks


def polygon_layer(obj, field, fill, opacity=1, stroke="#637d6e"):
    return f'<polygon points="{pts(obj[field])}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1"><title>{esc(obj.get("label", obj["id"]))}: {esc(obj.get("basis", ""))}</title></polygon>'


def topview(plan):
    t = plan["terrain"]
    items = ['<g id="map-content">']
    boundary = pts(t["plan_boundary"])
    items += [
        f'<defs><clipPath id="plan-clip"><polygon points="{boundary}"/></clipPath><pattern id="unresolved" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,8L8,0" stroke="#b7b4a4" stroke-width=".5"/></pattern></defs>',
        f'<polygon points="{boundary}" fill="#eee9d9" stroke="#989e90" stroke-width="1.2" stroke-dasharray="5 3"/>',
        f'<polygon points="{boundary}" fill="url(#unresolved)" opacity=".55"/>',
        '<g clip-path="url(#plan-clip)">',
    ]
    for cat in ("water", "mountain", "flat", "other"):
        items.append(f'<g class="layer-{cat}">')
        for o in t["terrain"]:
            if o["category"] != cat:
                continue
            items.append(polygon_layer(o, "plan_polygon", COLORS[cat]))
            if cat == "mountain":
                cx = sum(p[0] for p in o["plan_polygon"]) / len(o["plan_polygon"])
                cy = sum(p[1] for p in o["plan_polygon"]) / len(o["plan_polygon"])
                for scale in (0.76, 0.52, 0.28):
                    rings = [
                        [cx + (x - cx) * scale, cy + (y - cy) * scale] for x, y in o["plan_polygon"]
                    ]
                    items.append(
                        f'<polygon points="{pts(rings)}" fill="none" stroke="#7e937e" stroke-width=".7" opacity=".55"/>'
                    )
                items.append(text(cx, cy, o["id"], 10, "#526e5c", extra='text-anchor="middle"'))
        items.append("</g>")
    items.append('<g class="layer-flat">')
    for z in t["zones"]:
        items.append(polygon_layer(z, "plan_polygon", "#f5ebce", 0.96, "#bdb08a"))
        p = min(z["plan_polygon"], key=lambda p: p[1])
        items.append(text(p[0], p[1] - 4, z["id"] + " " + z["label"], 9, "#6d7459"))
    items.append('</g><g class="grid" opacity=".2">')
    for x in range(0, 1001, 25):
        items.append(
            f'<path d="M{x},0V1000 M0,{x}H1000" fill="none" stroke="#65776a" stroke-width=".45"/>'
        )
    items.append('</g><g class="layer-other">')
    for route in t["routes"]:
        points = pts(route["plan_points"])
        dash = ' stroke-dasharray="4 3"' if route["confidence"] == "low" else ""
        items.append(f'<polyline points="{points}" fill="none" stroke="#f4edd5" stroke-width="5"/>')
        items.append(
            f'<polyline points="{points}" fill="none" stroke="#8b8e73" stroke-width="1.3"{dash}><title>{esc(route["id"] + ": " + route["basis"])}</title></polyline>'
        )
    items.append('</g><g class="layer-tree">')
    for g in t["trees"]:
        for x, y in g["plan_roots"]:
            items.append(
                f'<g><title>{esc(g["id"] + " " + g["basis"])}</title><circle cx="{x}" cy="{y}" r="3" fill="#386e54" fill-opacity=".7"/><path d="M{x - 4},{y}h8 M{x},{y - 4}v8" stroke="#254f3b" stroke-width=".65"/></g>'
            )
    items.append('</g><g class="layer-house">')
    for b in plan["layout"]["buildings"]:
        corners = rect_corners(b)
        x, y = b["plan_center"]
        w, d = b["plan_size"]
        a = math.pi * b["front_clock"] / 6
        f = (math.sin(a), -math.cos(a))
        color = "#b68561" if b["category"] == "house" else "#849791"
        if b.get("grouped"):
            color = "#c5a17c"
        items.append(
            f'<g class="building" data-object="{esc(b["id"])}"><title>{esc(b["id"] + " " + b["label"] + " | " + b["assumption"])}</title>'
        )
        items.append(
            f'<polygon points="{pts(corners)}" fill="{color}" fill-opacity=".72" stroke="#77533f" stroke-width="1.2" stroke-dasharray="3 1.8"/>'
        )
        items.append(
            f'<path d="M{pts([corners[0]])} L{pts([corners[1]])}" fill="none" stroke="#623c27" stroke-width="2.2"/>'
        )
        ax, ay = x + f[0] * (d / 2 + 3), y + f[1] * (d / 2 + 3)
        bx, by = ax + f[0] * 7, ay + f[1] * 7
        items.append(
            f'<path d="M{ax},{ay}L{bx},{by}l{-f[0] * 3 - f[1] * 2},{-f[1] * 3 + f[0] * 2} M{bx},{by}l{-f[0] * 3 + f[1] * 2},{-f[1] * 3 - f[0] * 2}" stroke="#694f34" stroke-width="1" fill="none"/>'
        )
        items.append(
            text(
                x,
                y + 3,
                b["id"] + ("*" if b.get("grouped") else ""),
                9,
                "#252e25",
                extra='text-anchor="middle" font-weight="bold" paint-order="stroke" stroke="#fff5db" stroke-width="2"',
            )
        )
        items.append("</g>")
    items.append("</g>")
    for c in plan.get("pending_candidates", []):
        x, y = c["plan_marker"]
        items.append(
            f'<g class="building" data-object="{esc(c["id"])}"><title>{esc(c["id"] + ": 未确认候选，仅位置提示，不代表地基或占地范围")}</title><circle cx="{x}" cy="{y}" r="6" fill="#f2e7f3" stroke="#93669c" stroke-width="1.5" stroke-dasharray="2 2"/>'
            + text(x, y - 10, c["id"] + " ?", 10, "#714779", extra='text-anchor="middle"')
            + "</g>"
        )
    items.append("</g></g>")
    return "".join(items)


def source_overlay(plan, encoded, grid=True):
    t = plan["terrain"]
    items = [
        f'<image width="1000" height="1000" preserveAspectRatio="none" href="data:image/jpeg;base64,{base64.b64encode(encoded).decode()}"/>'
    ]
    items.append('<g class="source-terrain" opacity=".35">')
    for o in t["terrain"]:
        items.append(polygon_layer(o, "source_polygon", COLORS[o["category"]], 0.22))
    items.append("</g>")
    for z in t["zones"]:
        clip = "src-" + z["id"]
        poly = pts(z["source_polygon"])
        items.append(
            f'<defs><clipPath id="{esc(clip)}"><polygon points="{poly}"/></clipPath></defs>'
        )
        items.append(
            f'<polygon points="{poly}" fill="#e4dba2" fill-opacity=".05" stroke="#f3e19a" stroke-width=".75" stroke-dasharray="4 3"/>'
        )
        if grid:
            cx = sum(p[0] for p in z["source_polygon"]) / len(z["source_polygon"])
            cy = sum(p[1] for p in z["source_polygon"]) / len(z["source_polygon"])
            axes = []
            for a in z["ground_axes_image"]:
                length = math.hypot(*a)
                axes.append([a[0] / length, a[1] / length])
            items.append(
                f'<g class="source-grid" style="display:none" clip-path="url(#{esc(clip)})" opacity=".4" stroke="#e3edc6" stroke-width=".5">'
            )
            for axis, offset in ((axes[0], axes[1]), (axes[1], axes[0])):
                for k in range(-55, 56):
                    x, y = cx + k * 12 * offset[0], cy + k * 12 * offset[1]
                    items.append(
                        f'<path d="M{x - axis[0] * 1200},{y - axis[1] * 1200}L{x + axis[0] * 1200},{y + axis[1] * 1200}"/>'
                    )
            items.append("</g>")
    items.append('<g class="layer-tree">')
    for g in t["trees"]:
        for x, y in g["source_roots"]:
            items.append(
                f'<circle cx="{x}" cy="{y}" r="2.2" fill="#aee9b4" stroke="#305f45" stroke-width=".5"/>'
            )
    items.append('</g><g class="layer-house">')
    for b in plan["layout"]["buildings"]:
        items.append(
            f'<g class="building" data-object="{esc(b["id"])}"><title>{esc(b["id"] + " " + b["evidence"])}</title><polygon points="{pts(b["source_footprint"])}" fill="#ffd97a" fill-opacity=".17" stroke="#ffdd83" stroke-width="1.35" stroke-dasharray="3 2"/>'
        )
        x0, y0, x1, y1 = b["source_bbox"]
        items.append(
            f'<rect class="bbox" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#f5b5c5" stroke-width=".65" opacity=".5"/>'
        )
        x, y = b["source_anchor"]
        items.append(
            text(
                x,
                y - 4,
                b["id"],
                10,
                "#fff7d7",
                extra='text-anchor="middle" font-weight="bold" paint-order="stroke" stroke="#46382a" stroke-width="2.8"',
            )
        )
        items.append("</g>")
    items.append("</g>")
    current_ids = {b["id"] for b in plan["layout"]["buildings"]}
    excluded = {
        c["id"]: c
        for audit in plan.get("identity_audits", [])
        for c in audit["changes"]
        if c["action"] in {"reject", "merge"} and c["id"] not in current_ids
    }
    if excluded:
        items.append('<g class="excluded-observations" style="display:none">')
        for oid, c in excluded.items():
            x0, y0, x1, y1 = c["before"]["source_bbox"]
            reason = c.get("evidence", "Merged identity")
            items.append(
                f'<g><title>{esc(oid + ": " + reason)}</title><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#d476a1" stroke-width="1.5" stroke-dasharray="4 2"/>'
            )
            items.append(
                text(
                    x0,
                    y0 - 4,
                    oid + " × 非独立房屋",
                    9,
                    "#e4b0c9",
                    extra='paint-order="stroke" stroke="#46382a" stroke-width="2"',
                )
                + "</g>"
            )
        items.append("</g>")
    for c in plan.get("pending_candidates", []):
        x0, y0, x1, y1 = c["source_bbox"]
        items.append(
            f'<g class="building" data-object="{esc(c["id"])}"><title>{esc(c["uncertainty"])}</title><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#d099ec" stroke-width="1.5" stroke-dasharray="4 2"/>'
            + text(
                x0,
                y0 - 4,
                c["id"] + " ?",
                10,
                "#dfb8ef",
                extra='paint-order="stroke" stroke="#46382a" stroke-width="2"',
            )
            + "</g>"
        )
    return "".join(items)


def render_focus(plan, ids, path, crop=None):
    """Diagnostic crop selected by data IDs; not a scene-specific correction."""
    objects = [b for b in plan["layout"]["buildings"] if b["id"] in ids]
    if {b["id"] for b in objects} != set(ids):
        raise ValueError("Unknown focus object")
    sw, sh = plan["source"]["size"]
    x0 = max(0, min(b["source_bbox"][0] for b in objects) - 25)
    y0 = max(0, min(b["source_bbox"][1] for b in objects) - 25)
    x1 = min(1000, max(b["source_bbox"][2] for b in objects) + 25)
    y1 = min(1000, max(b["source_bbox"][3] for b in objects) + 25)
    if crop is None:
        crop = [
            int(x0 * sw / 1000),
            int(y0 * sh / 1000),
            math.ceil(x1 * sw / 1000),
            math.ceil(y1 * sh / 1000),
        ]
    data, meta = prepare_image(plan["source"]["path"], 2000, crop)
    w, h = meta["transmitted_size"]
    scale = w / (crop[2] - crop[0])
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 65}"><rect width="100%" height="100%" fill="#f4eddc"/>',
        text(12, 26, "候选重叠检查 · 框不是事实", 20),
        f'<image y="65" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(data).decode()}"/>',
    ]
    palette = ["#ffde69", "#6ce1f2", "#ff82b3", "#adf08b"]
    for i, b in enumerate(objects):
        u, v, uu, vv = b["source_bbox"]
        xx = (u * sw / 1000 - crop[0]) * scale
        yy = 65 + (v * sh / 1000 - crop[1]) * scale
        ww = (uu - u) * sw / 1000 * scale
        hh = (vv - v) * sh / 1000 * scale
        color = palette[i % len(palette)]
        svg.append(
            f'<rect x="{xx}" y="{yy}" width="{ww}" height="{hh}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            text(
                xx + 5,
                yy + 24,
                b["id"],
                22,
                color,
                'paint-order="stroke" stroke="#25352d" stroke-width="3"',
            )
        )
    svg.append("</svg>")
    rasterize("".join(svg), path)
    return path


def frame(content, title, subtitle, footer, width=1400, height=1510):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 1120 1210" font-family="Noto Sans CJK SC, sans-serif"><rect width="1120" height="1210" fill="#f6f2e7"/>{text(45, 44, title, 25)}{text(45, 73, subtitle, 12, "#69756a")}<g transform="translate(60 105)">{content}</g>{text(45, 1144, footer, 12, "#5c685d")}{text(45, 1176, "外缘仅为重建范围，非岸线；山地纹线非实测等高线。网格无米制尺度、真北或标定含义。", 11, "#7d8473")}</svg>'


def render(plan_path, png=False):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    out = path.parent
    source_path = Path(plan["source"]["path"])
    if not source_path.is_absolute():
        plan["source"]["path"] = str((path.parent / source_path).resolve())
    if (
        hashlib.sha256(Path(plan["source"]["path"]).read_bytes()).hexdigest()
        != plan["source"]["sha256"]
    ):
        raise ValueError("Source image changed since inference")
    encoded, _ = prepare_image(plan["source"]["path"], 2600)
    sw, sh = plan["source"]["size"]
    source_content = (
        f'<g transform="translate(0 {(1000 - 1000 * sh / sw) / 2}) scale(1 {sh / sw})">'
        + source_overlay(plan, encoded)
        + "</g>"
    )
    contents = {
        "topview": frame(
            topview(plan),
            "整幅俯视地基图 · 空间关系近似重建",
            "深色边 / 箭头：正面朝向　虚线：补全地基　圆点：树根代表位置　浅色斜纹：未定地面",
            "山地  灰绿　｜　局部平地 / 台地  米黄　｜　水域  蓝　｜　房屋  棕　｜　树根  绿",
        ),
        "source_overlay": frame(
            source_content,
            "原画对应 · 地基补全与局部地面网格",
            "粉框：建筑可见范围　黄虚线：推测底层地基　浅网格：分区地面方向假设",
            "编号与俯视图一一对应。遮挡补全不是原画已证实的事实；楼层、山体高度不直接投成地基。",
        ),
    }
    for name, svg in contents.items():
        ET.fromstring(svg)
        (out / (name + ".svg")).write_text(svg)
        if png:
            rasterize(svg, out / (name + ".png"))
    checks = geometry_checks(plan)
    counts = dict(Counter(b["category"] for b in plan["layout"]["buildings"]))
    save_data = {
        "building_counts": counts,
        "terrain_counts": dict(Counter(t["category"] for t in plan["terrain"]["terrain"])),
        "grouped_uncertain_buildings": [
            b["id"] for b in plan["layout"]["buildings"] if b.get("grouped")
        ],
        "pending_candidates_not_counted_as_buildings": [
            c["id"] for c in plan.get("pending_candidates", [])
        ],
        "tree_groups": len(plan["terrain"]["trees"]),
        "tree_root_symbols": sum(len(t["plan_roots"]) for t in plan["terrain"]["trees"]),
        "geometry_issues": checks,
        "no_metric_calibration": True,
        "no_local_gpu": True,
    }
    (out / "geometry_report.json").write_text(
        json.dumps(save_data, ensure_ascii=False, indent=2) + "\n"
    )
    cards = []
    for index, b in enumerate(plan["layout"]["buildings"]):
        ratio = b["plan_size"][0] / b["plan_size"][1]
        x0, y0, x1, y1 = b["source_bbox"]
        px = (x1 - x0) * 0.25
        py = (y1 - y0) * 0.3
        crop = [
            max(0, int((x0 - px) * sw / 1000)),
            max(0, int((y0 - py) * sh / 1000)),
            min(sw, math.ceil((x1 + px) * sw / 1000)),
            min(sh, math.ceil((y1 + py) * sh / 1000)),
        ]
        thumb, _ = prepare_image(plan["source"]["path"], 600, crop)
        thumbname = f"building_{index:03}.jpg"
        (out / thumbname).write_bytes(thumb)
        shape_note = "组包络比例，非单栋" if b.get("grouped") else "地基比例假设"
        cards.append(
            f'<tr id="row-{esc(b["id"])}"><td>{esc(b["id"])}<br>{esc(b["label"])}<br><a href="{thumbname}"><img src="{thumbname}" width="150" loading="lazy" alt="{esc(b["id"])} 原画局部"></a></td><td>{esc(b["subtype"])}<br>{"建筑组；栋数未定" if b.get("grouped") else ""}</td><td>{ratio:.2f}（{b["ratio_range"][0]}–{b["ratio_range"][1]}）<br>{shape_note}</td><td>{b["front_clock"]} 点钟</td><td>{esc(b["zone_id"])} / {esc(b["confidence"])}</td><td>{esc(b["evidence"])}<br><em>{esc(b["assumption"])}</em></td></tr>'
        )
    # Old reviewer prose may refer to superseded counts/ratios. The current
    # object records are authoritative; history remains available in plan.json.
    uncertainties = (
        plan["terrain"]["assumptions"]
        + [
            "建筑组用范围包络表达，不将其长宽比或单元范围解释为独栋实测结果。",
            "原画坐标与平面坐标分离；布局适配只改变平面位置。历史审查文字可能描述旧版本，当前数值以本表为准。",
        ]
        + [b["id"] + ": " + b["assumption"] for b in plan["layout"]["buildings"]]
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>古画 · 完整俯视地基图</title><style>body{margin:0;background:#e6e5d9;color:#263e33;font:15px/1.7 system-ui,sans-serif}header{padding:18px 28px;background:#263f34;color:#f3eddc;position:sticky;top:0;z-index:2}header p{margin:3px 0}button,label{margin-right:18px}main{padding:22px}h1{font-size:23px;margin:0}.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel svg{width:100%;height:auto}.panel{background:#f6f2e7}.building{cursor:pointer}.building:hover polygon,.building.selected polygon{stroke:#ec543f;stroke-width:3;fill:#ffd070}table{border-collapse:collapse;width:100%;background:#f6f2e7}td,th{border-bottom:1px solid #c4c9b8;padding:10px;text-align:left;vertical-align:top}tr.selected{background:#ffe3ba}em{color:#846a4c;font-style:normal}a{color:#507659}.notice{background:#f3e8cc;padding:12px}details{margin:20px 0}#selection{min-height:28px} @media(max-width:900px){.panels{grid-template-columns:1fr}header{position:static}}.source-grid{display:none}</style>
<header><h1>古画 → 完整俯视地基图</h1><p>共享相对坐标，保留局部高差与补全假设；不是测绘还原。点击任一建筑，两图同步高亮。</p><button id="solo" type="button">放大俯视图</button><label><input type="checkbox" id="grid" checked>俯视网格</label><label><input type="checkbox" id="sourcegrid">原画倾斜网格</label><label><input type="checkbox" id="roots" checked>树根</label><label><input type="checkbox" id="bbox" checked>原画范围框</label><div id="selection">选择建筑可查看编号、朝向与补全依据。</div></header><main>"""
    page += f'<p class="notice">当前含 {len(plan["layout"]["buildings"])} 个建筑/构筑物占地区，其中 {len(save_data["grouped_uncertain_buildings"])} 处为栋数未定的建筑组（*）；不能将占地区数量当成房屋准确总数。树林以 {save_data["tree_groups"]} 组代表根点记录，点数也不是树木准确数量。</p>'
    if plan.get("pending_candidates"):
        page += f"<p>另有 {len(plan['pending_candidates'])} 处紫色问号候选，尚不能确认是独立房屋；只标位置，不生成地基，也不计入上述数量。</p>"
    page += '<p><button id="source-solo" type="button">放大原画标注</button><label><input id="excluded" type="checkbox">显示本轮排除的旧框（不占地基）</label></p>'
    if plan.get("identity_audits"):
        removed = [
            c
            for a in plan["identity_audits"]
            for c in a["changes"]
            if c["action"] in {"reject", "merge"}
        ]
        page += (
            "<details><summary>本轮标注复核：排除/合并依据</summary><ul>"
            + "".join(
                "<li>"
                + esc(c["id"] + "：" + c.get("evidence", "归并到 " + str(c.get("merge_into"))))
                + "</li>"
                for c in removed
            )
            + "</ul><p>实物识别、原画观察与平面布局分别保存；不因平面避碰移动原画坐标。可见范围与推测地基不是同一件事。</p></details>"
        )
    if plan.get("layout_optimizations"):
        fit = plan["layout_optimizations"][-1]
        page += f"<p>布局整理仅移动了 {len(fit['changes'])} 个平面中心，最大允许位移 {fit['max_shift_relative_units']} 个相对单位；未修改识别范围、身份、长宽或朝向。这些位移仍是假设。</p>"
    page += (
        '<div class="panels"><div class="panel">'
        + contents["source_overlay"]
        + '</div><div class="panel">'
        + contents["topview"]
        + "</div></div>"
    )
    page += '<p><a href="topview.png">下载俯视 PNG</a> · <a href="topview.svg">矢量 SVG</a> · <a href="source_overlay.png">原画对应</a> · <a href="plan.json">完整数据与来源</a> · <a href="geometry_report.json">几何检查</a></p>'
    page += (
        "<details><summary>近似与未定事项</summary><ul>"
        + "".join("<li>" + esc(u) + "</li>" for u in uncertainties)
        + "</ul></details>"
    )
    page += (
        "<h2>建筑地基清单</h2><table><tr><th>实例</th><th>类型</th><th>前墙宽 / 进深</th><th>正面方向</th><th>支承区 / 置信度</th><th>证据与假设</th></tr>"
        + "".join(cards)
        + "</table>"
    )
    if plan.get("pending_candidates"):
        page += "<h2>待定候选（没有生成地基）</h2><table>"
        for index, c in enumerate(plan["pending_candidates"]):
            x0, y0, x1, y1 = c["source_bbox"]
            crop = [
                max(0, int((x0 - 25) * sw / 1000)),
                max(0, int((y0 - 25) * sh / 1000)),
                min(sw, math.ceil((x1 + 25) * sw / 1000)),
                min(sh, math.ceil((y1 + 25) * sh / 1000)),
            ]
            data, _ = prepare_image(plan["source"]["path"], 700, crop)
            name = f"pending_{index:03}.jpg"
            (out / name).write_bytes(data)
            page += f'<tr id="row-{esc(c["id"])}"><td>{esc(c["id"] + " ? " + c["label"])}<br><a href="{name}"><img src="{name}" width="220" alt="未定候选原画"/></a></td><td>{esc(c["evidence"])}<br><em>{esc(c["uncertainty"])}</em><p>紫色圆点只是相对位置提示，不是地基大小、类别确认或准确地图坐标。</p></td></tr>'
        page += "</table>"
    page += """<script>document.getElementById('solo').addEventListener('click',e=>{const active=e.target.dataset.active!=='yes';e.target.dataset.active=active?'yes':'no';e.target.textContent=active?'恢复原画对照':'放大俯视图';document.querySelector('.panels').style.gridTemplateColumns=active?'1fr':'1fr 1fr';document.querySelector('.panel').style.display=active?'none':'block';});</script>"""
    page += """<script>document.getElementById('source-solo').addEventListener('click',e=>{const panels=document.querySelectorAll('.panel');const active=e.target.dataset.active!=='yes';e.target.dataset.active=active?'yes':'no';e.target.textContent=active?'恢复双图':'放大原画标注';document.querySelector('.panels').style.gridTemplateColumns=active?'1fr':'1fr 1fr';panels[0].style.display='block';panels[1].style.display=active?'none':'block';const solo=document.getElementById('solo');solo.dataset.active='no';solo.textContent='放大俯视图';});document.getElementById('solo').addEventListener('click',()=>{document.querySelectorAll('.panel')[1].style.display='block';const source=document.getElementById('source-solo');source.dataset.active='no';source.textContent='放大原画标注';});document.getElementById('excluded').addEventListener('change',e=>document.querySelectorAll('.excluded-observations').forEach(n=>n.style.display=e.target.checked?'inline':'none'));</script>"""
    page += """</main><script>document.querySelectorAll('.building').forEach(el=>el.addEventListener('click',()=>{const id=el.dataset.object;document.querySelectorAll('.selected').forEach(n=>n.classList.remove('selected'));document.querySelectorAll('.building').forEach(n=>{if(n.dataset.object===id)n.classList.add('selected')});const row=document.getElementById('row-'+id);if(row){row.classList.add('selected');document.getElementById('selection').textContent=row.innerText.replaceAll('\\n',' · ')}}));for(const [id,cls] of [['grid','.grid'],['sourcegrid','.source-grid'],['roots','.layer-tree'],['bbox','.bbox']])document.getElementById(id).addEventListener('change',e=>document.querySelectorAll(cls).forEach(n=>n.style.display=e.target.checked?'inline':'none'));</script></html>"""
    (out / "index.html").write_text(page)
    print(
        json.dumps(
            {"report": str(out / "index.html"), "counts": counts, "geometry_issues": checks},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--png", action="store_true")
    a = p.parse_args()
    render(a.plan, a.png)
