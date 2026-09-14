"""Presentation-quality PNG cards drawn from real source images and plan data.

This renderer never changes recognition or footprint geometry. All decorative
lines are explicitly schematic. Outputs are 3840x2160 PNGs, with no slide deck.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path

from ancientplan.foundation2d.grounding import rasterize
from ancientplan.foundation2d.complete_plan import accepted, read
from ancientplan.foundation2d.plan_schema import validate_plan_stage
from ancientplan.foundation2d.qwen_cloud import prepare_image
from ancientplan.foundation2d.render_complete_plan import rect_corners, geometry_checks


INK = "#243f36"
MUTED = "#708077"
PAPER = "#f5f1e7"
ACCENT = "#a56f4a"


def text(x, y, value, size=24, fill=INK, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" {extra}>{html.escape(str(value))}</text>'


def points(poly):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in poly)


def draw_plan(plan):
    """Plot plan coordinates without converting roof bboxes into footprints."""
    terrain = plan["terrain"]
    boundary = points(terrain["plan_boundary"])
    parts = [
        f'<defs><clipPath id="map-clip"><polygon points="{boundary}"/></clipPath>'
        '<pattern id="ground-hatch" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M0,10L10,0" stroke="#d4d0c0" stroke-width=".55"/></pattern>'
        '<pattern id="water-lines" width="22" height="18" patternUnits="userSpaceOnUse"><path d="M2,9h8" stroke="#e8f0ee" stroke-width=".7"/></pattern></defs>',
        f'<polygon points="{boundary}" fill="#efebdf" stroke="#a6af9e" stroke-width="1.2" stroke-dasharray="4 4"/>',
        f'<polygon points="{boundary}" fill="url(#ground-hatch)" opacity=".45"/>',
        '<g clip-path="url(#map-clip)">',
    ]
    colors = {"water": "#b5d1d2", "mountain": "#b7c4ae", "flat": "#ede4c9", "other": "#d7d5c7"}
    for category in ("water", "mountain", "flat", "other"):
        for obj in terrain["terrain"]:
            if obj["category"] != category:
                continue
            poly = obj["plan_polygon"]
            parts.append(
                f'<polygon points="{points(poly)}" fill="{colors[category]}" stroke="#91a58f" stroke-width=".8"/>'
            )
            if category == "water":
                parts.append(f'<polygon points="{points(poly)}" fill="url(#water-lines)"/>')
            if category == "mountain":
                clip_id = "mountain-" + html.escape(obj["id"], quote=True)
                parts.append(
                    f'<defs><clipPath id="{clip_id}"><polygon points="{points(poly)}"/></clipPath></defs><g clip-path="url(#{clip_id})">'
                )
                cx, cy = [sum(p[k] for p in poly) / len(poly) for k in (0, 1)]
                for scale in (0.79, 0.59, 0.39):
                    ring = [[cx + (x - cx) * scale, cy + (y - cy) * scale] for x, y in poly]
                    parts.append(
                        f'<polygon points="{points(ring)}" fill="none" stroke="#8ba184" opacity=".6" stroke-width=".75"/>'
                    )
                parts.append("</g>")
    for zone in terrain["zones"]:
        parts.append(
            f'<polygon points="{points(zone["plan_polygon"])}" fill="#f2e8c9" fill-opacity=".96" stroke="#beaf88" stroke-width=".9"/>'
        )
    for x in range(0, 1001, 25):
        parts.append(
            f'<path d="M{x},0V1000 M0,{x}H1000" stroke="#748a79" stroke-width=".4" opacity=".13"/>'
        )
    for route in terrain["routes"]:
        pts = points(route["plan_points"])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#faf6e9" stroke-width="5"/>')
        dash = 'stroke-dasharray="3 3"' if route["confidence"] == "low" else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#9c9479" stroke-width="1.3" {dash}/>'
        )
    for group in terrain["trees"]:
        for x, y in group["plan_roots"]:
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="5.5" fill="#678d6b" fill-opacity=".22"/><circle cx="{x}" cy="{y}" r="2.6" fill="#426d51"/><path d="M{x - 4},{y}h8 M{x},{y - 4}v8" stroke="#466e52" stroke-width=".6"/>'
            )
    for building in plan["layout"]["buildings"]:
        corners = rect_corners(building)
        fill = "#b78058" if building["category"] == "house" else "#8aa39c"
        x, y = building["plan_center"]
        parts.append(
            f'<polygon points="{points(corners)}" fill="{fill}" fill-opacity=".7" stroke="#805b43" stroke-width="1.45" stroke-dasharray="3 1.6"/>'
        )
        parts.append(
            f'<polyline points="{points(corners[:2])}" fill="none" stroke="#704a35" stroke-width="2.8"/>'
        )
        direction = math.pi * building["front_clock"] / 6
        fx, fy = math.sin(direction), -math.cos(direction)
        reach = building["plan_size"][1] / 2
        ax, ay = x + fx * (reach + 3), y + fy * (reach + 3)
        bx, by = x + fx * (reach + 12), y + fy * (reach + 12)
        parts.append(
            f'<path d="M{ax},{ay}L{bx},{by}m{-fx * 3 - fy * 2},{-fy * 3 + fx * 2}L{bx},{by}l{-fx * 3 + fy * 2},{-fy * 3 - fx * 2}" fill="none" stroke="#704a35" stroke-width="1.3"/>'
        )
        label = building["id"] + ("*" if building.get("grouped") else "")
        # Explicit two-pass halo remains legible in CairoSVG and browser SVG.
        parts.append(
            text(
                x,
                y + 4,
                label,
                11,
                "none",
                'text-anchor="middle" font-weight="600" stroke="#fff9e8" stroke-width="3"',
            )
        )
        parts.append(text(x, y + 4, label, 11, "#334135", 'text-anchor="middle" font-weight="600"'))
    for candidate in plan.get("pending_candidates", []):
        x, y = candidate["plan_marker"]
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="7" fill="none" stroke="#9a7a9b" stroke-width="1.5" stroke-dasharray="2 2"/>'
        )
        parts.append(text(x, y - 11, candidate["id"] + " ?", 11, "#875c88", 'text-anchor="middle"'))
    parts.append("</g>")
    return "".join(parts)


def fit_box(width, height, x, y, box_width, box_height):
    factor = min(box_width / width, box_height / height)
    return (
        x + (box_width - width * factor) / 2,
        y + (box_height - height * factor) / 2,
        width * factor,
        height * factor,
    )


def content_bounds(plan):
    """Frame modeled ground rather than empty sky; do not alter coordinates."""
    terrain = plan["terrain"]
    pts = [p for key in ("terrain", "zones") for obj in terrain[key] for p in obj["plan_polygon"]]
    pts += [p for obj in terrain["trees"] for p in obj["plan_roots"]]
    pts += [p for obj in terrain["routes"] for p in obj["plan_points"]]
    pts += [p for b in plan["layout"]["buildings"] for p in rect_corners(b)]
    pts += [c["plan_marker"] for c in plan.get("pending_candidates", [])]
    pts = pts or terrain["plan_boundary"]
    return [min(p[k] for p in pts) - 24 for k in (0, 1)] + [
        max(p[k] for p in pts) + 24 for k in (0, 1)
    ]


def label_boxes(buildings, frame):
    """Greedy display-label spacing; observation anchors never move."""
    sx, sy, sw, sh = frame
    placed = []
    for building in buildings:
        ax, ay = building["source_anchor"]
        ax, ay = sx + ax * sw / 1000, sy + ay * sh / 1000
        width, height = 16 * len(building["id"]) + 12, 33
        candidates = []
        for dx, dy in (
            (0, -32),
            (0, 8),
            (-width - 8, -32),
            (width + 8, -32),
            (0, -76),
            (0, 52),
            (-width - 8, 8),
            (width + 8, 8),
        ):
            x = max(sx, min(sx + sw - width, ax - width / 2 + dx))
            y = max(sy, min(sy + sh - height, ay + dy))
            collisions = sum(
                max(0, min(x + width + 4, bx + bw) - max(x - 4, bx))
                * max(0, min(y + height + 4, by + bh) - max(y - 4, by))
                for bx, by, bw, bh in placed
            )
            candidates.append((collisions, dx * dx + (dy + 32) ** 2, x, y))
        _, _, x, y = min(candidates)
        placed.append((x, y, width, height))
    return placed


def card(row, plan, image_path, output):
    original = deepcopy(plan)
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != plan["source"]["sha256"]:
        raise ValueError("Source image does not match plan")
    data, meta = prepare_image(image_path, 3200)
    width, height = meta["transmitted_size"]
    sx, sy, sw, sh = fit_box(width, height, 175, 435, 1585, 1330)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="3840" height="2160" viewBox="0 0 3840 2160" font-family="Noto Sans CJK SC, sans-serif">',
        f'<rect width="3840" height="2160" fill="{PAPER}"/>',
        '<rect x="104" y="94" width="12" height="190" rx="5" fill="#365c4a"/>',
        text(150, 126, "CYANO TECH  /  ANCIENT PAINTING → PLAN", 24, MUTED, 'letter-spacing="4"'),
        text(147, 223, row["title"], 77, INK, 'font-weight="600"'),
        text(150, 276, row["theme"], 29, MUTED),
        text(
            3707,
            208,
            f"{row['sequence']:02}",
            115,
            "#cab895",
            'text-anchor="end" font-family="DejaVu Serif"',
        ),
        '<path d="M150,324H3690" stroke="#cecbbb" stroke-width="2"/>',
        '<rect x="125" y="370" width="1690" height="1450" rx="22" fill="#fffcf5" stroke="#ddd6c5" stroke-width="2"/>',
        '<rect x="2015" y="370" width="1690" height="1450" rx="22" fill="#fffcf5" stroke="#ddd6c5" stroke-width="2"/>',
        text(174, 417, "01  输入画作与地基编号", 27, INK),
        text(2064, 417, "02  相对地基俯视", 27, INK),
        f'<image x="{sx}" y="{sy}" width="{sw}" height="{sh}" href="data:image/jpeg;base64,{base64.b64encode(data).decode()}"/>',
    ]
    buildings = plan["layout"]["buildings"]
    labels = label_boxes(buildings, (sx, sy, sw, sh))
    for building, (lx, ly, label_width, label_height) in zip(buildings, labels):
        poly = [[sx + x * sw / 1000, sy + y * sh / 1000] for x, y in building["source_footprint"]]
        parts.append(
            f'<polygon points="{points(poly)}" fill="#ffe8a9" fill-opacity=".08" stroke="#edda9d" stroke-width="1.9" stroke-dasharray="8 5"/>'
        )
        x, y = building["source_anchor"]
        x, y = sx + x * sw / 1000, sy + y * sh / 1000
        label = building["id"]
        if abs(lx + label_width / 2 - x) > 5 or abs(ly + label_height - y) > 5:
            parts.append(
                f'<path d="M{x},{y}L{lx + label_width / 2},{ly + label_height / 2}" stroke="#efe0b5" stroke-width="1.8"/>'
            )
        parts.append(
            f'<rect x="{lx}" y="{ly}" width="{label_width}" height="{label_height}" rx="6" fill="#2f4c3d" fill-opacity=".93" stroke="#e9ddbd" stroke-width="1"/>'
        )
        parts.append(
            text(
                lx + label_width / 2,
                ly + 24,
                label,
                22,
                "#fff4d6",
                'text-anchor="middle" font-weight="500"',
            )
        )
    xmin, ymin, xmax, ymax = content_bounds(plan)
    mx, my, mw, mh = fit_box(xmax - xmin, ymax - ymin, 2065, 457, 1590, 1260)
    parts.append(
        f'<svg x="{mx}" y="{my}" width="{mw}" height="{mh}" viewBox="{xmin} {ymin} {xmax - xmin} {ymax - ymin}">'
        + draw_plan(plan)
        + "</svg>"
    )
    parts += [
        '<circle cx="1918" cy="1090" r="40" fill="#e7e4d8"/>',
        '<path d="M1900,1090h34m-13,-12l13,12l-13,12" fill="none" stroke="#70836f" stroke-width="3"/>',
    ]
    legend = [
        ("房屋地基", "#bd8e68"),
        ("其他占地", "#8aa39c"),
        ("局部平地", "#e8daba"),
        ("山地示意", "#acbea4"),
        ("水域", "#b5d1d2"),
        ("树根代表点", "#51785c"),
    ]
    for i, (label, color) in enumerate(legend):
        x = 2070 + i * 260
        parts.append(f'<rect x="{x}" y="1751" width="22" height="22" rx="3" fill="{color}"/>')
        parts.append(text(x + 34, 1772, label, 24, MUTED))
    buildings = plan["layout"]["buildings"]
    grouped = sum(bool(b.get("grouped")) for b in buildings)
    pending = len(plan.get("pending_candidates", []))
    house_count = sum(b["category"] == "house" for b in buildings)
    other_count = len(buildings) - house_count
    count_caption = f"{len(buildings):02} 个占地区域   ·   {house_count} 房屋 / {other_count} 其他"
    if grouped:
        count_caption += f"   ·   {grouped} 个未拆清组团 *"
    if pending:
        count_caption += f"   ·   {pending} 个未定候选"
    parts += [
        text(150, 1901, "从可见结构，到地面关系", 37, INK, 'font-weight="500"'),
        text(150, 1961, "屋顶 → 墙柱 → 支承地基；遮挡补全保留假设，不把图像框当占地。", 27, MUTED),
        text(3690, 1901, count_caption, 28, INK, 'text-anchor="end"'),
        text(
            3690,
            1961,
            "虚线为推测边界；箭头为正面方向；网格与地形纹线不表示实测尺度。",
            24,
            MUTED,
            'text-anchor="end"',
        ),
        '<path d="M150,2020H3690" stroke="#cecbbb" stroke-width="1.5"/>',
        text(
            150,
            2080,
            row.get("credit") or row.get("name", "Source attribution unavailable").split("_")[-1],
            23,
            MUTED,
        ),
        text(
            3690,
            2080,
            "QWEN 3.8  ·  2D FOUNDATION STUDY  ·  NON-METRIC",
            22,
            MUTED,
            'text-anchor="end" letter-spacing="2"',
        ),
        "</svg>",
    ]
    svg = "".join(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    rasterize(svg, output)
    if plan != original:
        raise AssertionError("Presentation rendering mutated inference geometry")
    return {
        "file": output.name,
        "size": [3840, 2160],
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "geometry_issues": geometry_checks(plan),
    }


def require_review(state, path, plan):
    """A ready flag alone cannot bless a changed or unrelated plan snapshot."""
    if not state.get("ready"):
        raise ValueError("Review incomplete")
    run = Path(state["final_check"]["run"])
    verdict = accepted(run)
    meta = read(run / "request_meta.json")
    context = read(meta["context_source"])
    if (
        meta["stage"] != "plan_final_check"
        or context["plan_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("Final review belongs to another plan snapshot")
    if verdict["verdict"] != "ready_for_confirmation" or validate_plan_stage(
        "plan_final_check", verdict, context
    ):
        raise ValueError("Final review did not accept this snapshot")
    if any(issue["severity"] == "major" for issue in geometry_checks(plan)):
        raise ValueError("Current geometry still has a major issue")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--allow-draft-preview", action="store_true")
    parser.add_argument(
        "--credits",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/example_sources.json",
    )
    args = parser.parse_args()
    if (
        args.allow_draft_preview
        and args.output.resolve() == Path(__file__).resolve().parents[1] / "examples"
    ):
        raise ValueError("Draft previews must stay outside the final examples directory")
    rows = json.loads((args.root / "selection.json").read_text())["examples"]
    credits = {r["slug"]: r for r in json.loads(args.credits.read_text())["examples"]}
    reports = []
    for row in rows:
        row = {**row, **credits[row["slug"]]}
        if args.ids and row["sequence"] not in args.ids:
            continue
        folder = args.root / row["directory"]
        state = json.loads((folder / "state.json").read_text())
        if not state.get("ready") and not args.allow_draft_preview:
            raise ValueError(f"Review incomplete: {row['directory']}")
        path = Path(state.get("reviewed_plan") or state["fitted_plan"])
        plan = json.loads(path.read_text())
        if not args.allow_draft_preview:
            require_review(state, path, plan)
        report = card(
            row,
            plan,
            folder / "source.jpg",
            args.output / f"{row['sequence']:02}_{row['slug']}.png",
        )
        reports.append(report)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
