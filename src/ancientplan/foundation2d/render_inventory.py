"""CPU SVG preview of cloud-returned visible-object boxes, not foundations."""

import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .grounding import rasterize
from .qwen_cloud import validate_inventory

COLORS = {
    "house": "#FF7966",
    "tree": "#79DFA0",
    "mountain": "#E5B678",
    "water": "#74CEF5",
    "flat": "#F4ED9C",
    "other": "#D1A5FF",
}
LABELS = {
    "house": "房屋",
    "tree": "树 / 树丛",
    "mountain": "山 / 岩石",
    "water": "水",
    "flat": "平地",
    "other": "其他",
}
FONT = "Noto Sans CJK SC, sans-serif"


def text(x, y, value, size=18, color="#E4EBE5"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-family="{FONT}">{html.escape(str(value))}</text>'


def render(run_dir, house_only=False):
    meta = json.loads((run_dir / "request_meta.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    normalized = run_dir / "inventory_normalized.json"
    display = json.loads(normalized.read_text()) if normalized.exists() else result
    data = display["parsed"]
    if (
        result.get("finish_reason") != "stop"
        or display["validation_issues"]
        or validate_inventory(data)
    ):
        raise ValueError(
            "Response has validation issues; do not silently repair or present it as complete."
        )
    iw, ih = meta["image"]["transmitted_size"]
    sw, sh = meta["image"]["source_size_after_exif"]
    is_crop = meta["image"]["crop_in_source_pixels"] != [0, 0, sw, sh]
    scope = "分区" if is_crop else "全图"
    width = 1200
    scale = width / iw
    height = round(ih * scale)
    total_height = max(height + 190, 1000)
    uri = "data:image/jpeg;base64," + base64.b64encode((run_dir / "input.jpg").read_bytes()).decode(
        "ascii"
    )
    f = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1580" height="{total_height}"><rect width="1580" height="{total_height}" fill="#17241E"/>',
        text(25, 43, f"Qwen 3.8 · {scope}可见元素首轮候选", 29),
        text(
            25,
            76,
            "这是对象外形框，不是地基；未经完整人工复核，灰区与小目标仍需放大巡查。",
            17,
            "#AFC6B9",
        ),
        f'<image x="25" y="105" width="{width}" height="{height}" href="{uri}"/>',
    ]
    objects = data["objects"]
    shown = [o for o in objects if not house_only or o["category"] == "house"]
    for obj in shown:
        a, b, c, d = obj["bbox"]
        x, y = 25 + a * width / 1000, 105 + b * height / 1000
        w, h = (c - a) * width / 1000, (d - b) * height / 1000
        col = COLORS[obj["category"]]
        dash = ' stroke-dasharray="5 3"' if obj.get("confidence") == "low" else ""
        f.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#13271D" stroke-width="3.4"/>'
        )
        f.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{col}" stroke-width="1.7"{dash}/>'
        )
        label = str(obj["id"])
        f.append(
            f'<rect x="{x}" y="{max(105, y - 20)}" width="{max(36, len(label) * 9 + 6)}" height="20" fill="#183529"/>'
        )
        f.append(text(x + 3, max(120, y - 5), label, 14, col))
    counts = Counter(o["category"] for o in objects)
    f.append(text(1250, 135, "云模型返回条目数", 20))
    for i, (category, label) in enumerate(LABELS.items()):
        f.append(text(1250, 179 + i * 42, f"{label}：{counts[category]}", 19, COLORS[category]))
    f += [
        text(1250, 469, f"合计：{len(objects)} 条", 20),
        text(1250, 509, "计数不是已验证真值", 16, "#E6C79D"),
        text(1250, 551, "树丛条目可能包含多株", 16, "#AFC6B9"),
        text(1250, 588, "只露一角仍可列为候选", 16, "#AFC6B9"),
        text(1250, 625, "虚线：模型低置信候选", 16, "#AFC6B9"),
        text(1250, 669, "未发送旧版人工标注", 16, "#AFC6B9"),
        text(1250, 706, "此图尚无遮挡补全", 16, "#AFC6B9"),
        text(1250, 749, "本机 CPU 绘图 / 无 GPU", 16, "#AFC6B9"),
        text(1250, 789, "树丛类别仅作格式归一", 16, "#AFC6B9"),
        text(1250, 825, "对象坐标与数量未修改", 16, "#AFC6B9"),
        text(
            25,
            total_height - 36,
            "仅显示房屋候选；完整类别见同目录 JSON 与全类别预览。"
            if house_only
            else "全类别框仅作首轮巡查索引；下一阶段需实例去重、局部放大和可见轮廓细化。",
            17,
            "#AFC6B9",
        ),
        "</svg>",
    ]
    svg = "".join(f)
    ET.fromstring(svg)
    name = "inventory_houses" if house_only else "inventory_all"
    (run_dir / (name + ".svg")).write_text(svg, encoding="utf-8")
    rasterize(svg, run_dir / (name + ".png"))
    print(run_dir / (name + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    render(args.run_dir)
    render(args.run_dir, True)
