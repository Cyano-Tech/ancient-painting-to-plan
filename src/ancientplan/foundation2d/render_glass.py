"""CPU-only, non-destructive vector layers for observed/inferred structure.

No coordinates or classifications are invented here. SVG/HTML show the exact
cloud proposal; an optional independent audit gates the default visible layers.
"""

import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .advance_scene import load_prior
from .scene_schema import validate_stage


LABELS = {
    "house": "房屋",
    "tree": "树木",
    "mountain": "山石",
    "water": "水",
    "flat": "平地",
    "other": "其他",
}


def esc(value):
    return html.escape(str(value), quote=True)


def svg_line(points, width, height, color, dashed=False, weight=2.5):
    coords = " ".join(
        f"{25 + x * width / 1000:.2f},{120 + y * height / 1000:.2f}" for x, y in points
    )
    dash = ' stroke-dasharray="9 6"' if dashed else ""
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{weight}" stroke-linecap="round" stroke-linejoin="round"{dash}/>'


def build_svg(run_dir, meta, structure, review, audit, mode):
    iw, ih = meta["image"]["transmitted_size"]
    width = 1400
    height = round(width * ih / iw)
    total_height = height + 210
    has_hidden = any(o["hidden_paths"] for o in structure["objects"])
    title = {
        "visible": "可见结构复核",
        "glass": "玻璃视图 · 隐藏结构假设" if has_hidden else "可见结构候选 · 本次未补隐藏线",
        "ground": "地面接触候选 · 尚未投影",
    }[mode]
    img = base64.b64encode((run_dir / "input.jpg").read_bytes()).decode("ascii")
    audit_by_id = {o["id"]: o for o in audit["objects"]} if audit else {}
    review_by_id = {o["id"]: o for o in review["objects"]}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="{total_height}" viewBox="0 0 1450 {total_height}">',
        "<style>.object:hover .shape{stroke-width:5}.object:hover .id{fill:white}.pending{display:none}</style>",
        f'<rect width="1450" height="{total_height}" fill="#14251f"/>',
        f'<text x="25" y="42" fill="#f2f2e8" font-family="Noto Sans CJK SC,sans-serif" font-size="28">Qwen 3.8 · {title}</text>',
        '<text x="25" y="76" fill="#c3d4c6" font-family="Noto Sans CJK SC,sans-serif" font-size="18">青实线：模型标注可见结构　粉虚线：推测隐藏部分　黄线：地面接触候选（虚线为推测）</text>',
        f'<image id="painting" x="25" y="120" width="{width}" height="{height}" opacity="{0.72 if mode == "glass" else 0.9}" href="data:image/jpeg;base64,{img}"/>',
    ]
    pending = 0
    for obj in structure["objects"]:
        oid = obj["id"]
        verdict = audit_by_id.get(oid)
        held = obj["status"] == "deferred" or (verdict and verdict["verdict"] != "plausible")
        pending += bool(held)
        classes = "object" + (" pending" if held else "")
        out.append(
            f'<g class="{classes}" data-id="{esc(oid)}" data-category="{esc(obj["category"])}">'
        )
        reason = "；".join(verdict.get("issues", [])) if verdict else "尚无独立模型复核"
        out.append(
            f"<title>{esc(oid)} {esc(LABELS[obj['category']])} | {esc(obj.get('assumption', ''))} | {esc(reason)}</title>"
        )
        out.append(f'<g class="visible" opacity="{0.28 if mode == "ground" else 1}">')
        for line in obj["visible_paths"]:
            out.append(svg_line(line, width, height, "#69f1da"))
        out.append(
            '</g><g class="hidden"' + (' style="display:none"' if mode != "glass" else "") + ">"
        )
        for line in obj["hidden_paths"]:
            out.append(svg_line(line, width, height, "#ff9ce0", True, 3))
        out.append(
            '</g><g class="footprint"' + (' style="display:none"' if mode != "ground" else "") + ">"
        )
        fp = obj["footprint"]
        points = fp["points"]
        safe = verdict and verdict.get("safe_for_ground_projection")
        # Audit-rejected ground contacts are never shown as accepted geometry.
        color = "#ffe28a" if safe else "#e79770"
        if fp["kind"] == "point":
            x, y = points[0]
            out.append(
                f'<circle cx="{25 + x * width / 1000}" cy="{120 + y * height / 1000}" r="6" fill="none" stroke="{color}" stroke-width="3"'
                + (' stroke-dasharray="3 2"' if fp["edge_status"][0] != "visible" else "")
                + "/>"
            )
        elif fp["kind"] in {"polygon", "polyline"}:
            for i, state in enumerate(fp["edge_status"]):
                out.append(
                    svg_line(
                        [points[i], points[(i + 1) % len(points)]],
                        width,
                        height,
                        color,
                        state != "visible",
                        3.2,
                    )
                )
        out.append("</g>")
        bbox = review_by_id[oid]["bbox"]
        x, y = 25 + bbox[0] * width / 1000, 120 + bbox[1] * height / 1000
        out.append(
            f'<g class="labels"><rect x="{x}" y="{y - 20}" width="58" height="22" rx="3" fill="#16322a"/><text class="id" x="{x + 4}" y="{y - 4}" fill="#f6eed3" font-family="sans-serif" font-size="15">{esc(oid)}</text></g></g>'
        )
    note = f"原图未修改；可见线也可能识别错误。默认隐藏 {pending} 个搁置/未通过复核对象；交互页可展开查看。"
    out.append(
        f'<text x="25" y="{height + 162}" fill="#d5d3b5" font-family="Noto Sans CJK SC,sans-serif" font-size="18">{esc(note)}</text>'
    )
    out.append(
        f'<text x="25" y="{height + 190}" fill="#b1c0b0" font-family="Noto Sans CJK SC,sans-serif" font-size="17">局部分区实验，不代表全图已完成；不是隐藏画面的真实恢复，也不是最终地基俯视图。</text></svg>'
    )
    svg = "".join(out)
    ET.fromstring(svg)
    return svg


def render(run_dir, audit_dir=None, png=False):
    run_dir = Path(run_dir).resolve()
    meta, data, structure_path = load_prior(run_dir)
    if meta["stage"] != "structure":
        raise ValueError("Expected a structure run")
    prior_path = Path(meta["context_source"])
    review = json.loads(prior_path.read_text())["parsed"]
    issues = validate_stage("structure", data, review)
    if issues:
        raise ValueError("Invalid structure: " + "; ".join(issues))
    audit = None
    if audit_dir:
        audit_meta, audit, _ = load_prior(audit_dir)
        if Path(
            audit_meta["context_source"]
        ).resolve() != structure_path.resolve() or validate_stage("audit", audit, data):
            raise ValueError("Audit does not match the structure")
    out_dir = run_dir / ("preview_audited" if audit else "preview_proposed")
    out_dir.mkdir(exist_ok=True)
    svgs = {}
    for mode in ("visible", "glass", "ground"):
        svg = build_svg(run_dir, meta, data, review, audit, mode)
        svgs[mode] = svg
        (out_dir / (mode + ".svg")).write_text(svg, encoding="utf-8")
        if png:
            from .grounding import rasterize

            rasterize(svg, out_dir / (mode + ".png"))
    audit_by_id = {o["id"]: o for o in audit["objects"]} if audit else {}
    rows = []
    for obj in data["objects"]:
        status = audit_by_id.get(obj["id"], {})
        rows.append(
            f"<tr><td>{esc(obj['id'])}</td><td>{esc(LABELS[obj['category']])}</td><td>{esc(obj['status'])}</td><td>{esc(obj['footprint']['kind'])}</td><td>{esc(status.get('verdict', '未复核'))}</td><td>{esc(obj.get('assumption', ''))}<br>{esc('；'.join(status.get('issues', [])))}</td></tr>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>古画结构玻璃视图</title>
<style>body{font:16px system-ui,sans-serif;background:#14251f;color:#e5ede2;margin:20px}header{position:sticky;top:0;background:#203c31;padding:12px;z-index:5;border-radius:8px}label{display:inline-block;margin:7px 16px 7px 0}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #486457;text-align:left}input{accent-color:#6de7c2}a{color:#9eeacb}.hint{color:#d8c99e}</style>
<header><b>原画上的玻璃视图 · 可核对的结构假设</b><br>
<label><input type="checkbox" checked data-layer="visible">可见结构</label>
<label><input type="checkbox" checked data-layer="hidden">隐藏结构假设</label>
<label><input type="checkbox" data-layer="footprint">地面接触候选</label>
<label><input type="checkbox" checked data-layer="labels">对象编号</label>
<label><input type="checkbox" id="pending">显示搁置/未通过复核对象</label>
<label>原画透明度 <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.72"></label><br>"""
    for category, label in LABELS.items():
        page += f'<label><input type="checkbox" checked data-category="{category}">{label}</label>'
    page += (
        '</header><p class="hint">青线是模型标注的可见结构，粉色虚线是假设；均可能有误。黄色地面候选通过本轮模型复核，橙色尚不宜投影。点击类别筛选，悬停对象查看说明。原图未改动，没有生成逼真隐藏纹理。</p>'
        + svgs["glass"]
    )
    page += (
        "<h2>逐对象证据与审查</h2><table><tr><th>ID</th><th>类别</th><th>结构状态</th><th>接地几何</th><th>模型复核</th><th>假设与问题</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    page += """<script>
function update(){for(const input of document.querySelectorAll('[data-layer]'))for(const node of document.querySelectorAll('svg .'+input.dataset.layer))node.style.display=input.checked?'':'none';
const showPending=document.getElementById('pending').checked;
for(const input of document.querySelectorAll('input[data-category]'))for(const node of document.querySelectorAll('svg .object[data-category="'+input.dataset.category+'"]'))node.style.display=input.checked&&(!node.classList.contains('pending')||showPending)?'inline':'none';
document.getElementById('painting').style.opacity=document.getElementById('opacity').value;}
document.querySelectorAll('input').forEach(i=>i.addEventListener('input',update));update();
</script></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "structure_run": str(run_dir),
                "audit_run": str(audit_dir) if audit_dir else None,
                "coordinate_system": "image_normalized_0_1000",
                "coordinates_changed": False,
                "local_model_used": False,
                "gpu_used": False,
                "coverage": "source crop only",
                "raw_object_count": len(data["objects"]),
                "structure_status_counts": dict(Counter(o["status"] for o in data["objects"])),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(out_dir / "index.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--png", action="store_true")
    a = p.parse_args()
    render(a.run_dir, a.audit, a.png)
