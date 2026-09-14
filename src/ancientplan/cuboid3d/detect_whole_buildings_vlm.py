#!/usr/bin/env python3
"""Propose complete building instances with open-source visual grounding.

The prompt defines architectural completeness, not any scene-specific count or
coordinates.  Output is a hash-bound proposal cache consumed by the audited
classifier/SAM pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .detect_building_instances import box_nms, file_sha256


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
WHOLE_BUILDING_PROMPT = """
Act as a visual object detector. Detect every distinct, complete house or
building visible in this artwork.

A valid building instance must have one coherent roof structure together with
the supporting wall/building body directly beneath it. Keep neighboring
buildings separate even when their roofs overlap. Include the entire visible
roof and visible supporting walls in one tight box.

Do not return standalone awnings, canopies, eaves, covered walks, roof-only
fragments, wall-only fragments, fences, gates, bridges, boats, rocks, trees,
water, or groups of multiple buildings. Do not guess an expected count.

Return JSON only, as an array in this exact form:
[{"bbox_1000":[x1,y1,x2,y2],"confidence":0.0}]
Coordinates are integers normalized to 0..1000 relative to the full image.
Return [] if there is no complete building.
""".strip()


def parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON array: {text}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("Grounding response is not a list")
    return value


def normalize_proposals(
    rows: list[dict],
    view_width: int,
    view_height: int,
    canvas_width: int,
    canvas_height: int,
    offset_x: int,
    offset_y: int,
    source: str,
) -> list[dict]:
    proposals = []
    for row in rows:
        box = row.get("bbox_1000")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (max(0.0, min(1000.0, float(value))) for value in box)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        pixel_box = [
            offset_x + x1 * view_width / 1000.0,
            offset_y + y1 * view_height / 1000.0,
            offset_x + x2 * view_width / 1000.0,
            offset_y + y2 * view_height / 1000.0,
        ]
        area_ratio = (
            (pixel_box[2] - pixel_box[0])
            * (pixel_box[3] - pixel_box[1])
            / max(canvas_width * canvas_height, 1)
        )
        if not 0.001 <= area_ratio <= 0.16:
            continue
        confidence = max(0.01, min(1.0, float(row.get("confidence", 0.5))))
        proposals.append(
            {
                "bbox": [round(value, 2) for value in pixel_box],
                "detector_score": round(confidence, 6),
                "proposal_source": f"qwen3_vl_whole_building_grounding:{source}",
            }
        )
    return proposals


def inference_views(
    image: Image.Image, include_tiles: bool
) -> list[tuple[str, Image.Image, int, int]]:
    views = [("full", image, 0, 0)]
    if not include_tiles:
        return views
    tile_width = round(image.width * 0.68)
    tile_height = round(image.height * 0.68)
    for row, y0 in enumerate((0, image.height - tile_height)):
        for column, x0 in enumerate((0, image.width - tile_width)):
            views.append(
                (
                    f"tile_r{row}_c{column}",
                    image.crop((x0, y0, x0 + tile_width, y0 + tile_height)),
                    x0,
                    y0,
                )
            )
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("whole_building_grounding"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--no-tiles", action="store_true", help="disable generic overlapping multiscale views"
    )
    parser.add_argument("--nms-threshold", type=float, default=0.58)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    proposals: list[dict] = []
    responses: list[dict] = []
    for source, view, offset_x, offset_y in inference_views(image, not args.no_tiles):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": view},
                    {"type": "text", "text": WHOLE_BUILDING_PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        responses.append({"source": source, "response": response})
        proposals.extend(
            normalize_proposals(
                parse_json_array(response),
                view.width,
                view.height,
                image.width,
                image.height,
                offset_x,
                offset_y,
                source,
            )
        )
    proposals = box_nms(proposals, args.nms_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.image.stem}_whole_buildings.json"
    overlay_path = args.output_dir / f"{args.image.stem}_whole_buildings.png"
    document = {
        "image": str(args.image.resolve()),
        "image_sha256": file_sha256(args.image),
        "model": args.model,
        "method": "open_source_vlm_architectural_completeness_grounding",
        "prompt": WHOLE_BUILDING_PROMPT,
        "raw_responses": responses,
        "multiscale_views": not args.no_tiles,
        "nms_threshold": args.nms_threshold,
        "proposals": proposals,
    }
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    for index, item in enumerate(proposals, 1):
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 36, 36, 255), width=4)
        label = f"WholeBuildingCandidate_{index:02d} {item['detector_score']:.2f}"
        bounds = draw.textbbox((x1 + 2, y1 + 2), label, font=font)
        draw.rectangle(bounds, fill=(12, 12, 12, 220))
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255, 255), font=font)
    overlay.save(overlay_path)
    print(
        json.dumps({"count": len(proposals), "json": str(json_path), "overlay": str(overlay_path)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
