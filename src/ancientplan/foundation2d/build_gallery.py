"""Build a small local gallery, not a claim of complete reconstruction."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

from .advance_scene import load_prior
from .qwen_cloud import SafeError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures", required=True, type=Path)
    p.add_argument("--audits", required=True, type=Path, nargs="+")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    structures = json.loads(a.structures.read_text())["tiles"]
    audits = [r for manifest in a.audits for r in json.loads(manifest.read_text())["tiles"]]
    audit_by_structure = {
        str(Path(r["prior_dir"]).resolve()): r for r in audits if r["exit_code"] == 0
    }
    a.output.mkdir(parents=True, exist_ok=True)

    def url(path):
        return html.escape(os.path.relpath(Path(path).resolve(), a.output.resolve()), quote=True)

    cards = []
    details = []
    for row in structures:
        run_dir = Path(row["output_dir"])
        # Raw exit status remains unchanged after a documented lossless format
        # normalization. Re-check the saved derivative rather than rewriting it.
        try:
            meta, data, _ = load_prior(run_dir)
        except (ValueError, OSError, SafeError):
            continue
        audit_row = audit_by_structure.get(str(run_dir.resolve()))
        if not audit_row:
            continue
        _, audit, _ = load_prior(audit_row["output_dir"])
        preview = run_dir / "preview_audited"
        if not (preview / "index.html").exists():
            raise ValueError("Render audited previews first")
        counts = Counter(o["category"] for o in data["objects"])
        verdicts = Counter(o["verdict"] for o in audit["objects"])
        hidden = sum(bool(o["hidden_paths"]) for o in data["objects"])
        polygons = sum(
            o["category"] == "house" and o["footprint"]["kind"] == "polygon"
            for o in data["objects"]
        )
        house_ids = {o["id"] for o in data["objects"] if o["category"] == "house"}
        ground_house_ids = [
            o["id"]
            for o in audit["objects"]
            if o["id"] in house_ids and o["safe_for_ground_projection"]
        ]
        detail = {
            "label": row["label"],
            "structure_run": str(run_dir.resolve()),
            "audit_run": audit_row["output_dir"],
            "candidate_counts": dict(counts),
            "objects_with_hidden_paths": hidden,
            "house_polygon_hypotheses": polygons,
            "audit_verdicts": dict(verdicts),
            "model_accepted_house_contact_ids": ground_house_ids,
            "crop_source_pixels": meta["image"]["crop_in_source_pixels"],
        }
        details.append(detail)
        title = html.escape(row["label"])
        card = f"<section><h2>{title}</h2><p>房屋候选 {counts['house']} · 原始隐藏线假设涉及 {hidden} 个对象（搁置项默认不显示） · 模型要求复核/搁置 {verdicts['needs_review'] + verdicts['reject']} 项。数量不是全图实例数。</p>"
        if not ground_house_ids:
            card += '<p class="warning">这一局部没有建筑接地候选通过本轮模型审查。结构线可用于讨论，不能直接生成建筑占地。</p>'
        card += f'<div class="pair"><figure><img src="{url(run_dir / "input.jpg")}"><figcaption>实际发送的原画局部</figcaption></figure><figure><a href="{url(preview / "index.html")}"><img src="{url(preview / "glass.png")}"></a><figcaption>结构预览；点击进入可切换图层</figcaption></figure></div>'
        card += f'<p><a href="{url(preview / "index.html")}">打开交互玻璃视图</a> · <a href="{url(preview / "ground.png")}">接地候选（非俯视图）</a> · <a href="{url(run_dir / "result.json")}">原始结构数据</a></p></section>'
        cards.append((counts["house"], row["label"], card))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图 3 · 局部结构实验</title>
<style>body{background:#172820;color:#ecede1;max-width:1480px;margin:32px auto;padding:0 24px;font:17px/1.6 system-ui,sans-serif}a{color:#9eecc4}section{background:#20372c;padding:20px;margin:24px 0;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;height:auto}figcaption{color:#c5cfbe;font-size:14px}.warning{border-left:4px solid #e5b375;padding:12px;background:#324132}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>图 3 · 局部结构与遮挡假设</h1>
<p>云端 Qwen 3.8-Max 识别与结构推断；本机 CPU 绘图，无本机 GPU 推理。</p>
<p class="warning">这是局部方法实验，不是全图标注成品或俯视图。旧候选只用于自动选裁剪范围，局部识别没有收到旧数量/旧解释。青实线为模型标注的可见线，粉虚线为推测。未通过结构审查的对象默认隐藏，但可以在交互页展开；通过模型审查也不等于真实正确。</p>
<p>原图和全部原始响应保留。遮挡补全不是恢复历史真相，局部接地候选还没有统一尺度或转换成俯视图。</p>"""
    page += "".join(c[2] for c in sorted(cards)) + "</html>"
    (a.output / "index.html").write_text(page, encoding="utf-8")
    (a.output / "gallery_manifest.json").write_text(
        json.dumps(
            {"cases": details, "global_map_complete": False, "local_gpu_used": False},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(a.output / "index.html")


if __name__ == "__main__":
    main()
