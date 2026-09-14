"""Render assumed foundations and local top-view alternatives; no global stitching."""

import argparse
import base64
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .advance_scene import load_prior
from .grounding import rasterize, validate_hypotheses, validate_ground_check
from .plane_geometry import hypothesis_grid


def esc(value):
    return html.escape(str(value), quote=True)


def label(x, y, value, size=16, color="#eaf0df"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="Noto Sans CJK SC,sans-serif" font-size="{size}">{esc(value)}</text>'


def svg_start(w, h):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#192d25"/>'
    ]


def render(run_dir, check_dir=None, png=False):
    run_dir = Path(run_dir).resolve()
    meta, data, data_path = load_prior(run_dir)
    if meta["stage"] != "ground_hypotheses":
        raise ValueError("Expected ground_hypotheses run")
    landmark_path = Path(meta["context_source"])
    landmark_data = json.loads(landmark_path.read_text())
    landmarks = landmark_data.get("parsed", landmark_data)
    if validate_hypotheses(data, landmarks):
        raise ValueError("Invalid ground hypotheses")
    decisions = {}
    checked = False
    corner_echo_checked = False
    if check_dir:
        cm, cd, _ = load_prior(check_dir)
        if (
            cm["stage"] != "ground_check"
            or Path(cm["context_source"]).resolve() != data_path.resolve()
            or validate_ground_check(
                cd, {"landmarks": landmarks, "hypotheses": data, **cm.get("context_flags", {})}
            )
        ):
            raise ValueError("Ground check does not match hypotheses")
        decisions = {(d["instance_id"], d["alternative_id"]): d for d in cd["decisions"]}
        checked = True
        corner_echo_checked = bool(cm.get("context_flags", {}).get("require_corner_echo"))
    w = 1200
    iw, ih = meta["image"]["transmitted_size"]
    h = round(w * ih / iw)
    overlay = svg_start(1240, h + 175)
    overlay += [
        label(20, 36, "地基位置假设 · 不是已确认占地", 25),
        label(
            20,
            68,
            "实线：模型引用的可见接地点　虚线：推测边　内部格子由假设生成，不证明视角正确",
            16,
            "#d7cba6",
        ),
        f'<image id="original" x="20" y="95" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode((run_dir / "input.jpg").read_bytes()).decode()}"/>',
    ]
    alternatives = [(o, a) for o in data["instances"] for a in o["alternatives"]]

    def is_accepted(pair):
        o, a = pair
        d = decisions.get((o["id"], a["id"]))
        return (
            d is not None
            and d["verdict"] == "usable_hypothesis"
            and (corner_echo_checked or a["placement"] == "shape_only")
        )

    if checked:
        alternatives.sort(key=lambda pair: not is_accepted(pair))
    visible_count = (
        sum(is_accepted(pair) for pair in alternatives) if checked else len(alternatives)
    )
    default_height = max(230, visible_count * 250 + 150)
    full_height = max(230, len(alternatives) * 250 + 150)
    plans = svg_start(1100, default_height)
    plans[0] = plans[0].replace(
        "<svg ",
        f'<svg id="topview_panel" data-default-height="{default_height}" data-full-height="{full_height}" ',
        1,
    )
    plans += [
        label(20, 36, "局部俯视形状候选 · 每行独立，不是共同坐标总图", 24),
        label(
            20,
            70,
            "同一方案左右展示长宽比区间两端；进深统一为相对单位 1，并非实测米数。",
            16,
            "#d7cba6",
        ),
    ]
    records = []
    rows = []
    for index, (obj, alt) in enumerate(alternatives):
        oid, aid = obj["id"], alt["id"]
        decision = decisions.get((oid, aid))
        accepted = (
            decision is not None
            and decision["verdict"] == "usable_hypothesis"
            and (corner_echo_checked or alt["placement"] == "shape_only")
        )
        pending = checked and not accepted
        css = ' class="proposal pending"' if pending else ' class="proposal"'
        display = ' style="display:none"' if pending else ""
        verdict = decision["verdict"] if decision else "not_checked"
        if decision and decision["verdict"] == "usable_hypothesis" and not accepted:
            verdict = "requires_corner_echo_check"
        color = "#ffe69a" if accepted else "#f5a2cf"
        record = {
            "instance_id": oid,
            "alternative_id": aid,
            "placement": alt["placement"],
            "verdict": verdict,
            "ratio_range": alt["width_depth_ratio_range"],
            "camera_calibration_verified": False,
            "plane_maps": [],
        }
        if alt["placement"] != "shape_only":
            points = alt["image_corners"]
            coords = " ".join(f"{20 + x * w / 1000},{95 + y * h / 1000}" for x, y in points)
            overlay.append(
                f'<g{css}{display} data-proposal="{esc(aid)}"><title>{esc(alt["placement_basis"])} | {esc(alt["uncertainty"])}</title>'
            )
            overlay.append(f'<polygon points="{coords}" fill="{color}" fill-opacity=".09"/>')
            for i, state in enumerate(alt["edge_status"]):
                a, b = points[i], points[(i + 1) % 4]
                dash = ' stroke-dasharray="8 6"' if state == "inferred" else ""
                overlay.append(
                    f'<line x1="{20 + a[0] * w / 1000}" y1="{95 + a[1] * h / 1000}" x2="{20 + b[0] * w / 1000}" y2="{95 + b[1] * h / 1000}" stroke="{color}" stroke-width="3"{dash}/>'
                )
            mid = sum(alt["width_depth_ratio_range"]) / 2
            grid, mapping = hypothesis_grid(points, mid)
            record["plane_maps"].append(mapping)
            overlay.append('<g class="grid" opacity=".38">')
            for a, b in grid:
                overlay.append(
                    f'<line x1="{20 + a[0] * w / 1000}" y1="{95 + a[1] * h / 1000}" x2="{20 + b[0] * w / 1000}" y2="{95 + b[1] * h / 1000}" stroke="{color}" stroke-width="1"/>'
                )
            overlay.append("</g>")
            for i, (x, y) in enumerate(points):
                overlay.append(
                    label(
                        25 + x * w / 1000, 88 + y * h / 1000, f"{oid}/{aid}:{'ABCD'[i]}", 13, color
                    )
                )
            overlay.append("</g>")
        y = 110 + index * 250
        plans.append(f"<g{css}{display}>")
        status = "仅形状，未定位" if alt["placement"] == "shape_only" else "位置也含假设"
        plans.append(label(20, y, f"{oid}/{aid} · {status} · {verdict}", 18, color))
        low, high = alt["width_depth_ratio_range"]
        scale = min(120, 420 / high)
        for j, ratio in enumerate((low, high)):
            x = 45 + j * 550
            fy = y + 35
            fw, fh = ratio * scale, scale
            plans.append(
                f'<rect x="{x}" y="{fy}" width="{fw}" height="{fh}" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-width="2" stroke-dasharray="7 4"/>'
            )
            for k in range(1, 4):
                plans.append(
                    f'<line x1="{x}" y1="{fy + fh * k / 4}" x2="{x + fw}" y2="{fy + fh * k / 4}" stroke="{color}" stroke-opacity=".25"/>'
                )
            for k in range(1, min(120, int(ratio * 4) + 1)):
                gx = x + k * scale / 4
                if gx < x + fw:
                    plans.append(
                        f'<line x1="{gx}" y1="{fy}" x2="{gx}" y2="{fy + fh}" stroke="{color}" stroke-opacity=".25"/>'
                    )
            plans.append(label(x, fy + fh + 24, f"前墙宽/进深 = {ratio:g}；前墙为上边", 16, color))
        plans.append("</g>")
        records.append(record)
        reason = decision["reason"] if decision else "未经独立检查"
        rows.append(
            f"<tr><td>{esc(oid)}/{esc(aid)}</td><td>{esc(alt['placement'])}</td><td>{low:g}–{high:g}</td><td>{esc(verdict)}</td><td>{esc(alt['placement_basis'])}<br>{esc(alt['ratio_basis'])}<br>{esc(alt['uncertainty'])}<hr>{esc(reason)}</td></tr>"
        )
    overlay.append(
        label(
            20,
            h + 134,
            "仅使用当前局部原画。没有统一各区域尺度、支承面或高差，不能据此拼接为全图。",
            16,
            "#d7cba6",
        )
    )
    overlay.append(
        label(
            20,
            h + 161,
            "坐标映射与格子由假设四边形构造；拟合误差小不代表识别或测量准确。",
            16,
            "#d7cba6",
        )
    )
    overlay.append("</svg>")
    if not alternatives:
        plans.append(label(20, 140, "本区域没有足够约束提出地基形状候选。", 20))
    plans.append("</svg>")
    osvg, psvg = "".join(overlay), "".join(plans)
    ET.fromstring(osvg)
    ET.fromstring(psvg)
    output = run_dir / ("foundation_checked" if checked else "foundation_proposed")
    output.mkdir(exist_ok=True)
    for name, svg in (("overlay", osvg), ("topview", psvg)):
        (output / (name + ".svg")).write_text(svg)
        if png:
            rasterize(svg, output / (name + ".png"))
    if not checked:
        (run_dir / "hypotheses_overlay.svg").write_text(osvg)
        if png:
            rasterize(osvg, run_dir / "hypotheses_overlay.png")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>地基与局部俯视假设</title><style>body{background:#192d25;color:#e9eddc;margin:24px;font:16px/1.6 system-ui,sans-serif}header{position:sticky;top:0;background:#284334;z-index:9;padding:15px}svg{width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{border:1px solid #526750;padding:10px;text-align:left}label{margin-right:20px}.warning{color:#e7ca9d}a{color:#a5e5bd}</style>
<header><b>地基候选 → 带假设标签的局部俯视图</b><br><label><input id="pending" type="checkbox">显示未通过方案</label><label><input id="grid" type="checkbox" checked>显示假设派生网格</label><label>原画透明度 <input id="opacity" type="range" min="0" max="1" step=".05" value="1"></label></header>
<p class="warning">这不是已完成的全图俯视图。二维形状以“地基近似矩形”和模型给出的长宽比区间为条件；内部格子是计算结果，不是独立证据。纯形状候选不会放进原画坐标。模型审查通过也不代表真实准确。</p>"""
    page += (
        osvg
        + "<h2>俯视形状范围（每行独立）</h2>"
        + psvg
        + "<h2>依据与审查</h2><table><tr><th>对象/方案</th><th>定位类型</th><th>宽/深</th><th>检查</th><th>依据与局限</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    page += """<script>function update(){const show=document.getElementById('pending').checked;document.querySelectorAll('.pending').forEach(n=>n.style.display=show?'inline':'none');document.querySelectorAll('.grid').forEach(n=>n.style.display=document.getElementById('grid').checked?'inline':'none');document.getElementById('original').style.opacity=document.getElementById('opacity').value;const panel=document.getElementById('topview_panel');const height=show?panel.dataset.fullHeight:panel.dataset.defaultHeight;panel.setAttribute('height',height);panel.setAttribute('viewBox','0 0 1100 '+height);}document.querySelectorAll('input').forEach(n=>n.addEventListener('input',update));update();</script></html>"""
    (output / "index.html").write_text(page)
    (output / "projection_manifest.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "check_run": str(check_dir) if check_dir else None,
                "global_map_complete": False,
                "camera_calibration_verified": False,
                "corner_echo_checked": corner_echo_checked,
                "units": "relative depth=1 independently for each instance",
                "records": records,
                "numerical_mapping_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(output / "index.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--check", type=Path)
    p.add_argument("--png", action="store_true")
    a = p.parse_args()
    render(a.run_dir, a.check, a.png)
