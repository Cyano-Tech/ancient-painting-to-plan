"""Code-native before/after annotation comparison on the identical source crop."""

import argparse
import base64
from pathlib import Path

from .complete_plan import read
from .grounding import rasterize
from .identity_audit import audit_crop
from .qwen_cloud import prepare_image
from .render_complete_plan import text


def render_comparison(before_path, after_path, ids, output):
    before, after = read(before_path), read(after_path)
    if before["source"] != after["source"]:
        raise ValueError("Different source images")
    old = {b["id"]: b for b in before["layout"]["buildings"]}
    new = {b["id"]: b for b in after["layout"]["buildings"]}
    if not set(ids) <= old.keys():
        raise ValueError("Unknown comparison ID")
    source = before["source"]
    sw, sh = source["size"]
    crop = audit_crop([old[i] for i in ids], source["size"])
    data, meta = prepare_image(source["path"], 1600, crop)
    w, h = meta["transmitted_size"]
    scale = w / (crop[2] - crop[0])
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * 2 + 60}" height="{h + 135}" font-family="Noto Sans CJK SC, sans-serif"><rect width="100%" height="100%" fill="#f6f2e7"/>'
    ]
    for index, (objects, title_) in enumerate(
        ((old, "修订前 · 旧候选框"), (new, "修订后 · 实体与非实体分开"))
    ):
        left = 20 + index * (w + 20)
        top = 65
        svg.append(text(left, 35, title_, 22))
        svg.append(
            f'<image x="{left}" y="{top}" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(data).decode()}"/>'
        )
        for oid in ids:
            b = objects.get(oid, old[oid])
            removed = oid not in objects
            x0, y0, x1, y1 = b["source_bbox"]
            x = left + (x0 * sw / 1000 - crop[0]) * scale
            y = top + (y0 * sh / 1000 - crop[1]) * scale
            width = (x1 - x0) * sw / 1000 * scale
            height = (y1 - y0) * sh / 1000 * scale
            color = "#f28bb2" if removed else "#8ee6d0" if index else "#f8d76c"
            dash = ' stroke-dasharray="7 4"' if removed else ""
            svg.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>'
            )
            label = oid + (" × 非独立实体" if removed else "")
            svg.append(
                text(
                    x + 3,
                    y + 22,
                    label,
                    18,
                    color,
                    'paint-order="stroke" stroke="#263b30" stroke-width="2.5"',
                )
            )
        svg.append(
            text(
                left,
                h + 96,
                "粉色虚框：排除，不生成另一份地基" if index else "框为当时识别结果，不是人工真值",
                15,
            )
        )
    svg.append("</svg>")
    output = Path(output)
    output.with_suffix(".svg").write_text("".join(svg))
    rasterize("".join(svg), output)
    print(output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render_comparison(a.before, a.after, a.ids, a.output)
