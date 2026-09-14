"""Publish local evidence and conditional top-view outputs without claiming a full map."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

from .advance_scene import load_prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landmarks", required=True, type=Path)
    p.add_argument("--hypotheses", required=True, type=Path)
    p.add_argument("--check", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    hm, hd, hypothesis_path = load_prior(a.hypotheses)
    cm, cd, _ = load_prior(a.check)
    if not cm.get("context_flags", {}).get("require_corner_echo"):
        raise ValueError("Primary example must have explicit corner echo check")
    if Path(cm["context_source"]).resolve() != hypothesis_path.resolve():
        raise ValueError("Primary check refers to a different hypothesis source")
    preview = a.hypotheses / "foundation_checked"
    if not (preview / "index.html").exists():
        raise ValueError("Render checked foundation first")
    a.output.mkdir(parents=True, exist_ok=True)

    def url(path):
        return html.escape(os.path.relpath(Path(path).resolve(), a.output.resolve()), quote=True)

    rows = []
    summaries = []
    for row in json.loads(a.landmarks.read_text())["tiles"]:
        run_dir = Path(row["output_dir"])
        meta, data, _ = load_prior(run_dir)
        points = [p for i in data["instances"] for p in i["landmarks"]]
        counts = dict(Counter(p["level"] for p in points))
        merges = [i["source_ids"] for i in data["instances"] if len(i["source_ids"]) > 1]
        summaries.append(
            {
                "label": row["label"],
                "run": str(run_dir.resolve()),
                "instances": len(data["instances"]),
                "point_level_counts": counts,
                "merged_source_groups": merges,
                "rejected": data["rejected"],
            }
        )
        rows.append(
            f'<tr><td>{html.escape(row["label"])}</td><td>{len(data["instances"])}</td><td>{html.escape(str(counts))}</td><td><a href="{url(run_dir / "landmarks.png")}">逐点图</a> · <a href="{url(run_dir / "result.json")}">原始结果</a></td></tr>'
        )
    accepted = [d for d in cd["decisions"] if d["verdict"] == "usable_hypothesis"]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图 3 · 地基与局部俯视候选</title><style>body{background:#17291f;color:#e7eddf;max-width:1500px;margin:30px auto;padding:0 24px;font:17px/1.6 system-ui,sans-serif}a{color:#a4edc2}section{background:#233a2d;padding:22px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}.pair{display:grid;grid-template-columns:1.1fr 1fr;gap:22px}figure{margin:0}figcaption{font-size:15px;color:#c9d4bd}.warning{border-left:4px solid #e4bd76;padding:14px;background:#334331}table{border-collapse:collapse;width:100%}th,td{border:1px solid #526c50;padding:12px;text-align:left}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>图 3 · 第一处地基与局部俯视候选</h1>
<p class="warning">这是一栋建筑的条件性俯视示意，不是完成的全图。地基四角含推测，真实长宽、后墙位置和支承面尚未测得；地基近似矩形是建模假设。地面网格由候选四边形推导，不能反过来证明它准确。</p>"""
    page += f'<section><h2>原画地基网格 ↔ 俯视形状范围</h2><p>通过本轮显式角点核对的方案：{len(accepted)} 个。未通过方案在交互页默认隐藏。</p><div class="pair"><figure><img src="{url(preview / "overlay.png")}"><figcaption>原画上的候选地基；虚线为推测边，实线也只是模型判断。</figcaption></figure><figure><img src="{url(preview / "topview.png")}"><figcaption>左右是宽/深区间两端，不能当作两栋建筑或实测尺寸。</figcaption></figure></div><p><a href="{url(preview / "index.html")}">打开可切换图层的检查页</a> · <a href="{url(a.check / "result.json")}">角点回显与审查原文</a></p></section>'
    page += "<section><h2>这轮具体改动</h2><p>将接地点、遮挡交界、楼层/檐口点分开；用原画与同尺寸坐标辅助图读点。发现短侧墙被错误当作前墙后，由云模型真正重排四角，再要求检查返回原样坐标及前墙索引，避免文字解释偷偷换角。</p><p>这里的坐标辅助网格是平直图像刻度，不是地面方向。上方展示的地基网格才是按候选地基方向变形后的网格。</p></section>"
    page += (
        "<section><h2>三个局部的逐点记录</h2><p>下表均是模型判断与候选数，不是已验证实例数；跨局部对象可能重复。</p><table><tr><th>局部</th><th>实例候选</th><th>点类别计数</th><th>检查资料</th></tr>"
        + "".join(rows)
        + "</table></section>"
    )
    page += '<p>仍未完成：全图唯一实例合并、全部类别地基、跨局部尺度/高差对齐与总俯视图。全部视觉推断来自云端 Qwen 3.8-Max，本机仅 CPU 预处理/绘图。</p><p><a href="PROGRESS.md">进展、用量与已知局限</a></p></html>'
    (a.output / "index.html").write_text(page)
    (a.output / "report_manifest.json").write_text(
        json.dumps(
            {
                "primary_hypothesis_run": str(a.hypotheses.resolve()),
                "primary_check_run": str(a.check.resolve()),
                "landmark_summaries": summaries,
                "accepted_hypotheses": accepted,
                "global_map_complete": False,
                "dimensions_measured": False,
                "gpu_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(a.output / "index.html")


if __name__ == "__main__":
    main()
