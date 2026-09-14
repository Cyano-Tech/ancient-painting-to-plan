#!/usr/bin/env python3
"""Infer each building's long-axis orientation and visible short-end evidence.

This is deliberately a second semantic pass.  It consumes the accepted
whole-building locations from the detector, and it never re-detects, adds, or
removes buildings.  The resulting cache is hash-bound to both the painting
and the detector document so stale locations cannot silently drive geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

from .detect_building_instances import file_sha256


DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
SEMANTIC_ORIENTATION_PROMPT = """
The first vision stage has already detected one complete building. You receive
an isolated crop containing only the SAME accepted SAM region on a neutral
background. Do not re-detect, re-box, merge, or split the target. Coordinates
below are normalized to this entire isolated crop: top-left is [0,0] and
bottom-right is [1000,1000].

Analyze the target's ground-plane LONG AXIS before choosing any clock angle.
The long axis runs along the roof ridge and long eaves, from one SHORT end wall
to the other. Do not mistake roof-slope lines (ridge to eave), a broad long
facade, an awning, fence, gate, vegetation, shadow, or a neighboring roof for
the long axis.

First trace the dominant ridge/long-eave family through the target. Then locate
the visible short END WALL supported below its roof/gable end. The requested
directed axis points from the footprint's ground-plane center toward the
ground-contact midpoint of that visible short end wall. Do not point to the
gable apex, a roof corner, the broad facade, or the image-box center. If part of
the wall base is occluded, infer its midpoint from the two wall corners and
vertical supports. The endpoint's lower screen y means nearer on the ground.

Also locate the TWO ground-contact corners at the bottom of that SAME short end
wall. They are the endpoints of the short wall's footprint edge, not the ends
of the long facade. Use the vertical wall posts/edges to carry each corner down
to the inferred ground plane when foliage, a fence, or shadow hides the base.
Never use the roof ridge endpoints, gable apex, eave tips, or two corners from
different walls. If the two base corners genuinely cannot be distinguished,
return null for the corner pair instead of inventing a very narrow wall.

Return JSON only with keys "id", "building_ground_center_1000" ([x,y]),
"visible_short_end_ground_midpoint_1000" ([x,y]), "visual_cues" (a short
string naming the ridge, end wall, and supporting corners),
"visible_short_end_ground_corners_1000" ([[x1,y1],[x2,y2]] or null),
"short_end_corners_confidence" (0 to 1), and "confidence" (0 to 1). Do not
return a direction letter or clock angle; geometry will derive those from the
independently grounded points.
""".strip()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_array(text: str) -> list[dict]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON array: {text}")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("Orientation response is not a list")
    return value


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON object: {text}")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Orientation response is not an object")
    return value


def parse_clock(value: object) -> float:
    match = re.fullmatch(r"\s*(\d{1,2})(?::(00|30))?\s*", str(value))
    if not match:
        raise ValueError(f"invalid half-hour clock value: {value!r}")
    hour = int(match.group(1)) % 12
    if hour == 0:
        hour = 12
    result = hour + (0.5 if match.group(2) == "30" else 0.0)
    return result


def clock_label(clock_hour: float) -> str:
    hour = int(math.floor(clock_hour))
    minute = 30 if abs(clock_hour - hour - 0.5) < 1e-6 else 0
    return f"{hour}:{minute:02d}"


def highlighted_crop(image: Image.Image, instance: dict) -> tuple[Image.Image, dict]:
    x1, y1, x2, y2 = map(float, instance["mask_bbox"])
    box_width, box_height = x2 - x1, y2 - y1
    pad_x = max(40.0, box_width * 0.35)
    pad_y = max(40.0, box_height * 0.45)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(image.width, int(math.ceil(x2 + pad_x)))
    bottom = min(image.height, int(math.ceil(y2 + pad_y)))
    crop = image.crop((left, top, right, bottom)).convert("RGBA")
    draw = ImageDraw.Draw(crop, "RGBA")
    local_box = (x1 - left, y1 - top, x2 - left, y2 - top)
    line_width = max(3, round(min(crop.size) / 110))
    draw.rectangle(local_box, outline=(0, 238, 255, 255), width=line_width)
    font = ImageFont.load_default()
    label = instance["id"]
    bounds = draw.textbbox((local_box[0] + 4, local_box[1] + 4), label, font=font)
    draw.rectangle(
        (bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2),
        fill=(0, 30, 34, 225),
    )
    draw.text((local_box[0] + 4, local_box[1] + 4), label, fill=(255, 255, 255, 255), font=font)
    crop = crop.convert("RGB")
    if max(crop.size) < 900:
        scale = 900.0 / max(crop.size)
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return crop, {
        "source_crop_bbox": [left, top, right, bottom],
        "target_bbox_crop_normalized": [
            (x1 - left) / max(right - left, 1),
            (y1 - top) / max(bottom - top, 1),
            (x2 - left) / max(right - left, 1),
            (y2 - top) / max(bottom - top, 1),
        ],
    }


def isolated_target_crop(
    image: Image.Image,
    instance: dict,
    output_size: tuple[int, int],
    metadata: dict,
) -> Image.Image:
    """Composite only the accepted SAM region onto a neutral background."""
    left, top, right, bottom = metadata["source_crop_bbox"]
    clean = image.crop((left, top, right, bottom)).convert("RGB")
    mask = Image.new("L", clean.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    for polygon in instance.get("visible_region_polygons", []):
        local_polygon = [(float(x) - left, float(y) - top) for x, y in polygon]
        if len(local_polygon) >= 3:
            mask_draw.polygon(local_polygon, fill=255)
    clean = clean.resize(output_size, Image.Resampling.LANCZOS)
    mask = mask.resize(output_size, Image.Resampling.NEAREST)
    neutral = Image.new("RGB", output_size, (226, 223, 208))
    return Image.composite(clean, neutral, mask)


def semantic_analysis_sheet(context_crop: Image.Image, isolated_crop: Image.Image) -> Image.Image:
    header_height = 34
    width = context_crop.width + isolated_crop.width
    height = max(context_crop.height, isolated_crop.height) + header_height
    sheet = Image.new("RGB", (width, height), (226, 223, 208))
    sheet.paste(context_crop, (0, header_height))
    sheet.paste(isolated_crop, (context_crop.width, header_height))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, header_height), fill=(36, 41, 38))
    draw.text((12, 11), "ORIGINAL CONTEXT (cyan target)", fill="white", font=font)
    draw.text(
        (context_crop.width + 12, 11),
        "SAME TARGET ISOLATED BY ACCEPTED MASK",
        fill="white",
        font=font,
    )
    return sheet


def direction_candidates(
    image: Image.Image,
    instance: dict,
    camera_pitch_deg: float,
    camera_azimuth_deg: float,
) -> list[dict]:
    """Build camera-rectified candidates from real line segments in the SAM mask."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for polygon in instance.get("visible_region_polygons", []):
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
    mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.bitwise_and(cv2.Canny(enhanced, 60, 150), mask)
    x1, y1, x2, y2 = map(int, instance["mask_bbox"])
    roi = edges[y1:y2, x1:x2]
    lines = cv2.HoughLinesP(
        roi,
        1,
        math.pi / 360.0,
        threshold=max(10, round((x2 - x1) / 28)),
        minLineLength=max(10, round((x2 - x1) / 28)),
        maxLineGap=max(4, round((x2 - x1) / 65)),
    )
    if lines is None:
        raise ValueError(f"{instance['id']}: no structural direction candidates inside SAM mask")

    pitch = math.radians(camera_pitch_deg)
    azimuth = math.radians(camera_azimuth_deg)
    groups: dict[float, dict] = {}
    for wrapped in lines:
        xa, ya, xb, yb = map(float, wrapped.reshape(-1)[:4])
        xa, xb = xa + x1, xb + x1
        ya, yb = ya + y1, yb + y1
        screen_dx, screen_dy = xb - xa, yb - ya
        length = math.hypot(screen_dx, screen_dy)
        if length < 10.0:
            continue
        # Resolve the undirected line to the visible/near half-plane. For an
        # exactly horizontal line, painting-right is the canonical endpoint.
        if screen_dy < 0 or (abs(screen_dy) < 1e-6 and screen_dx < 0):
            xa, ya, xb, yb = xb, yb, xa, ya
            screen_dx, screen_dy = -screen_dx, -screen_dy
        screen_h = screen_dx
        far_h = -screen_dy / max(math.sin(pitch), 1e-6)
        world_x = math.cos(azimuth) * screen_h + math.sin(azimuth) * far_h
        world_y = -math.sin(azimuth) * screen_h + math.cos(azimuth) * far_h
        top_angle = math.degrees(math.atan2(-world_y, world_x)) % 360.0
        if top_angle > 180.0:
            top_angle = (top_angle + 180.0) % 360.0
            xa, ya, xb, yb = xb, yb, xa, ya
        quantized_angle = min(180.0, max(0.0, round(top_angle / 15.0) * 15.0))
        group = groups.setdefault(
            quantized_angle,
            {"support": 0.0, "representative_score": -1.0, "source_line_px": None},
        )
        group["support"] += length
        if length > group["representative_score"]:
            group["representative_score"] = length
            group["source_line_px"] = [round(xa, 2), round(ya, 2), round(xb, 2), round(yb, 2)]

    candidates = []
    # Use a fixed, exhaustive half-hour vocabulary.  Hough evidence is kept as
    # provenance/debug information, but is deliberately not allowed to remove
    # a direction before the semantic model sees it.
    for index, angle in enumerate(np.arange(0.0, 360.0, 15.0)):
        axis_angle = float(angle % 180.0)
        evidence_options = [groups.get(axis_angle)]
        if axis_angle == 0.0:
            evidence_options.append(groups.get(180.0))
        evidence = max(
            (item for item in evidence_options if item is not None),
            key=lambda item: item["support"],
            default=None,
        )
        if evidence is None:
            evidence = {
                "support": 0.0,
                "representative_score": -1.0,
                "source_line_px": None,
            }
        clock_hour, clock = clock_from_top_view_angle(angle)
        candidates.append(
            {
                "candidate_id": chr(ord("A") + index),
                "top_view_angle_deg_clockwise_from_right": angle,
                "clock_hour": clock_hour,
                "clock": clock,
                "source_line_px": evidence["source_line_px"],
                "line_support": round(evidence["support"], 3),
            }
        )
    return candidates


def candidate_sheet(
    crop: Image.Image,
    metadata: dict,
    candidates: list[dict],
    camera_pitch_deg: float,
) -> Image.Image:
    columns, rows = 4, math.ceil(len(candidates) / 4)
    panel_width, panel_height, header_height = 360, 250, 30
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), (232, 229, 216))
    font = ImageFont.load_default()
    left, top, right, bottom = metadata["source_crop_bbox"]
    for index, candidate in enumerate(candidates):
        panel = Image.new("RGB", (panel_width, panel_height), (244, 241, 228))
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rectangle((0, 0, panel_width, header_height), fill=(36, 41, 38, 255))
        draw.text(
            (12, 9),
            f"Candidate {candidate['candidate_id']}",
            fill=(255, 255, 255, 255),
            font=font,
        )
        fitted = crop.copy()
        fitted.thumbnail(
            (panel_width - 10, panel_height - header_height - 10), Image.Resampling.LANCZOS
        )
        paste_x = (panel_width - fitted.width) // 2
        paste_y = header_height + (panel_height - header_height - fitted.height) // 2
        panel.paste(fitted, (paste_x, paste_y))
        target_box = metadata["target_bbox_crop_normalized"]
        center_x = paste_x + (target_box[0] + target_box[2]) * 0.5 * fitted.width
        center_y = paste_y + (target_box[1] + target_box[3]) * 0.5 * fitted.height
        ground_angle = math.radians(candidate["top_view_angle_deg_clockwise_from_right"])
        screen_dx = math.cos(ground_angle)
        screen_dy = math.sin(ground_angle) * math.sin(math.radians(camera_pitch_deg))
        norm = max(math.hypot(screen_dx, screen_dy), 1e-6)
        half_length = min(fitted.width, fitted.height) * 0.25
        screen_dx, screen_dy = screen_dx / norm * half_length, screen_dy / norm * half_length
        origin_x, origin_y = center_x - screen_dx, center_y - screen_dy
        end_x, end_y = center_x + screen_dx, center_y + screen_dy
        draw.line((origin_x, origin_y, end_x, end_y), fill=(210, 30, 28, 245), width=7)
        arrow_angle = math.atan2(end_y - origin_y, end_x - origin_x)
        for wing in (-2.5, 2.5):
            draw.line(
                (
                    end_x,
                    end_y,
                    end_x + math.cos(arrow_angle + wing) * 13,
                    end_y + math.sin(arrow_angle + wing) * 13,
                ),
                fill=(210, 30, 28, 245),
                width=5,
            )
        column, row = index % columns, index // columns
        sheet.paste(panel, (column * panel_width, row * panel_height))
    return sheet


def normalized_point(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be one [x,y] point")
    try:
        x, y = (max(0.0, min(1000.0, float(component))) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} has invalid coordinates: {value!r}") from exc
    return x, y


def normalized_optional_point_pair(
    value: object,
    field_name: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be two [x,y] points or null")
    first = normalized_point(value[0], f"{field_name}[0]")
    second = normalized_point(value[1], f"{field_name}[1]")
    if math.dist(first, second) < 3.0:
        return None
    return first, second


def normalized_confidence(value: object, default: float = 0.5) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"high", "medium", "low"}:
        value = {"high": 0.9, "medium": 0.6, "low": 0.3}[value.strip().lower()]
    return max(0.0, min(1.0, float(value)))


def clock_from_top_view_angle(angle_deg: float) -> tuple[float, str]:
    quantized_angle = (round((angle_deg % 360.0) / 15.0) * 15.0) % 360.0
    clock_hour = (3.0 + quantized_angle / 30.0) % 12.0
    if clock_hour <= 0.5:
        clock_hour += 12.0
    return clock_hour, clock_label(clock_hour)


def normalize_rows(
    rows: list[dict],
    expected_ids: list[str],
    candidates_by_building: dict[str, list[dict]],
) -> list[dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("id")) not in expected_ids:
            continue
        building_id = str(row["id"])
        if building_id in by_id:
            raise ValueError(f"duplicate orientation row for {building_id}")
        candidate_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates_by_building[building_id]
        }
        candidate_id = re.sub(
            r"^CANDIDATE\s+",
            "",
            str(row.get("candidate_id", "")).strip().upper(),
        )
        if candidate_id not in candidate_by_id:
            raise ValueError(f"{building_id}: invalid direction candidate {candidate_id!r}")
        candidate = candidate_by_id[candidate_id]
        horizontal = str(row.get("visible_side_horizontal", "")).strip().lower()
        depth = str(row.get("visible_side_depth", "")).strip().lower()
        if horizontal not in {"left", "right"}:
            raise ValueError(
                f"{building_id}: invalid visible-side horizontal relation {horizontal!r}"
            )
        if depth not in {"nearer", "same_depth", "farther"}:
            raise ValueError(f"{building_id}: invalid visible-side depth relation {depth!r}")
        angle = math.radians(candidate["top_view_angle_deg_clockwise_from_right"])
        if abs(math.cos(angle)) >= 0.2:
            candidate_horizontal = "right" if math.cos(angle) > 0 else "left"
            if horizontal != candidate_horizontal:
                raise ValueError(f"{building_id}: candidate contradicts its left/right self-check")
        candidate_depth = (
            "nearer"
            if math.sin(angle) > 0.2
            else "farther"
            if math.sin(angle) < -0.2
            else "same_depth"
        )
        if depth != candidate_depth:
            raise ValueError(f"{building_id}: candidate contradicts its near/far self-check")
        confidence = normalized_confidence(row.get("confidence"))
        by_id[building_id] = {
            "id": building_id,
            "candidate_id": candidate_id,
            "visible_side_horizontal": horizontal,
            "visible_side_depth": depth,
            "clock": candidate["clock"],
            "clock_hour": candidate["clock_hour"],
            "top_view_angle_deg_clockwise_from_right": candidate[
                "top_view_angle_deg_clockwise_from_right"
            ],
            "raw_camera_rectified_angle_deg": row.get("raw_camera_rectified_angle_deg"),
            "orientation_candidate_score": row.get("orientation_candidate_score"),
            "orientation_line_support": row.get("orientation_line_support"),
            "grounded_alignment_deg": row.get("grounded_alignment_deg"),
            "building_ground_center_source_px": row.get("building_ground_center_source_px"),
            "visible_side_ground_midpoint_source_px": row.get(
                "visible_side_ground_midpoint_source_px"
            ),
            "visible_short_end_ground_corners_source_px": row.get(
                "visible_short_end_ground_corners_source_px"
            ),
            "short_end_corners_confidence": round(
                normalized_confidence(row.get("short_end_corners_confidence"), 0.0), 6
            ),
            "short_end_corner_midpoint_residual_px": row.get(
                "short_end_corner_midpoint_residual_px"
            ),
            "visual_cues": row.get("visual_cues", ""),
            "confidence": round(confidence, 6),
        }
    missing = [building_id for building_id in expected_ids if building_id not in by_id]
    if missing:
        raise ValueError(f"orientation response omitted accepted IDs: {missing}")
    return [by_id[building_id] for building_id in expected_ids]


def draw_orientation_overlay(
    image: Image.Image, instances: list[dict], orientations: list[dict]
) -> Image.Image:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    by_id = {row["id"]: row for row in orientations}
    for instance in instances:
        row = by_id[instance["id"]]
        x1, y1, x2, y2 = map(float, instance["mask_bbox"])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        start_px = row.get("building_ground_center_source_px")
        end_px = row.get("visible_side_ground_midpoint_source_px")
        end_corners = row.get("visible_short_end_ground_corners_source_px")
        if start_px and end_px:
            draw.line((*start_px, *end_px), fill=(70, 255, 105, 245), width=5)
            radius = 6
            draw.ellipse(
                (end_px[0] - radius, end_px[1] - radius, end_px[0] + radius, end_px[1] + radius),
                fill=(70, 255, 105, 255),
            )
        if end_corners:
            first, second = end_corners
            draw.line((*first, *second), fill=(255, 214, 52, 255), width=6)
            for corner in (first, second):
                radius = 6
                draw.ellipse(
                    (
                        corner[0] - radius,
                        corner[1] - radius,
                        corner[0] + radius,
                        corner[1] + radius,
                    ),
                    fill=(255, 214, 52, 255),
                )
        angle = math.radians(row["top_view_angle_deg_clockwise_from_right"])
        length = max(45.0, min(140.0, (x2 - x1) * 0.36))
        dx, dy = math.cos(angle) * length, math.sin(angle) * length
        draw.rectangle((x1, y1, x2, y2), outline=(0, 238, 255, 230), width=3)
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(255, 48, 35, 255), width=7)
        # Arrowhead marks the VLM-inferred visible short-end direction.
        end_x, end_y = cx + dx, cy + dy
        wing = 14.0
        for offset in (-2.55, 2.55):
            draw.line(
                (
                    end_x,
                    end_y,
                    end_x + math.cos(angle + offset) * wing,
                    end_y + math.sin(angle + offset) * wing,
                ),
                fill=(255, 48, 35, 255),
                width=5,
            )
        label = f"{instance['id']}  {row['clock']}"
        bounds = draw.textbbox((x1 + 4, y1 + 4), label, font=font)
        draw.rectangle(
            (bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2), fill=(10, 10, 10, 225)
        )
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255, 255), font=font)
    return overlay.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("detection_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("building_orientation"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        required=True,
        help="Torch device, or 'auto' to shard the orientation model across available GPUs",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=int,
        default=27,
        help="per-visible-GPU weight budget used by accelerate auto placement",
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=18.0,
        help="effective ground-plane elevation for oblique/cavalier painting rectification",
    )
    parser.add_argument("--camera-azimuth-deg", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--awq-backend",
        default="torch_awq",
        help="portable GPTQModel backend used only for AWQ checkpoints",
    )
    parser.add_argument(
        "--reuse-grounding-json",
        type=Path,
        help="reuse hash-bound raw VLM point responses and rerun only deterministic geometry",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    detection = json.loads(args.detection_json.read_text(encoding="utf-8"))
    if detection.get("stage") != "automatic_whole_building_instance_parsing":
        raise ValueError("orientation stage requires the accepted whole-building detector document")
    if tuple(detection.get("image_size", ())) != image.size:
        raise ValueError("detector image size does not match orientation image")
    instances = detection.get("instances", [])
    expected_ids = [str(instance["id"]) for instance in instances]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("detector document has no instances or duplicate IDs")

    reused_responses: dict[str, str] = {}
    if args.reuse_grounding_json:
        reused_document = json.loads(args.reuse_grounding_json.read_text(encoding="utf-8"))
        if reused_document.get("image_sha256") != file_sha256(args.image):
            raise ValueError("reused grounding document is bound to a different image")
        if reused_document.get("detection_sha256") != file_hash(args.detection_json):
            raise ValueError("reused grounding document is bound to a different detector result")
        if reused_document.get("model") != args.model:
            raise ValueError("reused grounding document was produced by a different model")
        reused_responses = {
            str(item["id"]): str(item["semantic_response"])
            for item in reused_document.get("raw_responses", [])
        }
        if set(reused_responses) != set(expected_ids):
            raise ValueError("reused grounding document does not contain exactly the accepted IDs")

    processor = None
    model = None
    input_device = None
    thinking_mode = "Thinking" in args.model
    if not reused_responses:
        processor = AutoProcessor.from_pretrained(args.model)
        model_kwargs: dict = {"attn_implementation": "sdpa", "dtype": "auto"}
        if args.model.endswith("-AWQ"):
            # Qwen2.5-VL AWQ leaves the vision tower unquantized. Transformers 5
            # nests that tower under ``model`` while the official legacy
            # checkpoint stores it at the root, so both the skip paths and the
            # explicit checkpoint-key mapping are required.
            model_config = AutoConfig.from_pretrained(args.model)
            model_config.torch_dtype = torch.float16
            model_config.dtype = torch.float16
            quantization_config = dict(model_config.quantization_config)
            quantization_config["backend"] = args.awq_backend
            quantization_config["modules_to_not_convert"] = ["visual", "model.visual"]
            model_config.quantization_config = quantization_config
            model_kwargs["config"] = model_config
            model_kwargs["dtype"] = torch.float16
            model_kwargs["key_mapping"] = {r"^visual\.": "model.visual."}
            model_kwargs["offload_buffers"] = True
        if args.device == "auto":
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {
                **{
                    index: f"{args.max_gpu_memory_gib}GiB"
                    for index in range(torch.cuda.device_count())
                },
                "cpu": "48GiB",
            }
        else:
            model_kwargs["device_map"] = {"": args.device}
        model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
        if args.model.endswith("-AWQ"):
            # The portable AWQ kernels preserve the checkpoint's BF16 hidden-state
            # stream, while Transformers casts the unquantized output head to FP16.
            # Restore the head to the stream dtype to avoid a final matmul mismatch.
            model.lm_head.to(dtype=torch.bfloat16)
        model.eval()
        input_device = next(model.parameters()).device

    def generate_response(messages: list[dict]) -> str:
        if processor is None or model is None or input_device is None:
            raise RuntimeError("model generation requested while reusing cached grounding")
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if thinking_mode:
            template_kwargs["enable_thinking"] = True
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(input_device)
        generation_kwargs = {"max_new_tokens": args.max_new_tokens}
        if thinking_mode:
            generation_kwargs.update(
                {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            )
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_responses: list[dict] = []
    rows: list[dict] = []
    candidates_by_building: dict[str, list[dict]] = {}
    candidate_sheet_paths: list[dict] = []
    semantic_sheet_paths: list[dict] = []
    for instance in instances:
        building_id = instance["id"]
        crop, metadata = highlighted_crop(image, instance)
        isolated = isolated_target_crop(image, instance, crop.size, metadata)
        analysis_sheet = semantic_analysis_sheet(crop, isolated)
        analysis_path = args.output_dir / f"{args.image.stem}_{building_id}_semantic_target.png"
        analysis_sheet.save(analysis_path)
        semantic_sheet_paths.append({"id": building_id, "path": str(analysis_path.resolve())})
        candidates = direction_candidates(
            image,
            instance,
            args.camera_pitch_deg,
            args.camera_azimuth_deg,
        )
        candidates_by_building[building_id] = candidates
        sheet = candidate_sheet(isolated, metadata, candidates, args.camera_pitch_deg)
        sheet_path = args.output_dir / f"{args.image.stem}_{building_id}_orientation_candidates.png"
        sheet.save(sheet_path)
        candidate_sheet_paths.append({"id": building_id, "path": str(sheet_path.resolve())})

        semantic_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": isolated},
                    {
                        "type": "text",
                        "text": SEMANTIC_ORIENTATION_PROMPT
                        + f"\n\nThe supplied target ID is {building_id}.",
                    },
                ],
            }
        ]
        semantic_response = reused_responses.get(building_id)
        if semantic_response is None:
            semantic_response = generate_response(semantic_messages)
        try:
            semantic_row = parse_json_object(semantic_response)
        except (ValueError, json.JSONDecodeError):
            print(f"Raw semantic response for {building_id}:\n{semantic_response}")
            raise
        try:
            center_1000 = normalized_point(
                semantic_row.get("building_ground_center_1000"),
                "building_ground_center_1000",
            )
            endpoint_1000 = normalized_point(
                semantic_row.get("visible_short_end_ground_midpoint_1000"),
                "visible_short_end_ground_midpoint_1000",
            )
            end_corners_1000 = normalized_optional_point_pair(
                semantic_row.get("visible_short_end_ground_corners_1000"),
                "visible_short_end_ground_corners_1000",
            )
            crop_width = metadata["source_crop_bbox"][2] - metadata["source_crop_bbox"][0]
            crop_height = metadata["source_crop_bbox"][3] - metadata["source_crop_bbox"][1]
            screen_dx = (endpoint_1000[0] - center_1000[0]) * crop_width / 1000.0
            screen_dy = (endpoint_1000[1] - center_1000[1]) * crop_height / 1000.0
            if math.hypot(screen_dx, screen_dy) < 3.0:
                raise ValueError(f"{building_id}: grounded orientation points are coincident")
            pitch = math.radians(args.camera_pitch_deg)
            azimuth = math.radians(args.camera_azimuth_deg)
            screen_h = screen_dx
            far_h = -screen_dy / max(math.sin(pitch), 1e-6)
            world_x = math.cos(azimuth) * screen_h + math.sin(azimuth) * far_h
            world_y = -math.sin(azimuth) * screen_h + math.cos(azimuth) * far_h
            top_angle = math.degrees(math.atan2(-world_y, world_x)) % 360.0
            horizontal_sign = 1.0 if screen_dx >= 0.0 else -1.0
            side_candidates = [
                item
                for item in candidates
                if math.cos(math.radians(item["top_view_angle_deg_clockwise_from_right"]))
                * horizontal_sign
                >= -1e-6
                and math.sin(math.radians(item["top_view_angle_deg_clockwise_from_right"]))
                * screen_dy
                >= -1e-6
            ]
            max_line_support = max(
                (float(item["line_support"]) for item in side_candidates),
                default=0.0,
            )
            scored_candidates = []
            for item in side_candidates:
                item_angle = item["top_view_angle_deg_clockwise_from_right"]
                angle_difference = abs((item_angle - top_angle + 180.0) % 360.0 - 180.0)
                support_score = float(item["line_support"]) / max(max_line_support, 1.0)
                grounding_score = math.cos(math.radians(angle_difference))
                combined_score = support_score + 1.1 * grounding_score
                scored_candidates.append(
                    (combined_score, -angle_difference, float(item["line_support"]), item)
                )
            candidate_score, _, _, candidate = max(scored_candidates, key=lambda item: item[:3])
            grounded_alignment_deg = abs(
                (candidate["top_view_angle_deg_clockwise_from_right"] - top_angle + 180.0) % 360.0
                - 180.0
            )
            angle = math.radians(candidate["top_view_angle_deg_clockwise_from_right"])
            end_corners_source_px = None
            corner_midpoint_residual_px = None
            if end_corners_1000 is not None:
                end_corners_source_px = [
                    [
                        round(metadata["source_crop_bbox"][0] + point[0] * crop_width / 1000.0, 2),
                        round(metadata["source_crop_bbox"][1] + point[1] * crop_height / 1000.0, 2),
                    ]
                    for point in end_corners_1000
                ]
                corner_midpoint = (
                    (end_corners_source_px[0][0] + end_corners_source_px[1][0]) / 2.0,
                    (end_corners_source_px[0][1] + end_corners_source_px[1][1]) / 2.0,
                )
                endpoint_source = (
                    metadata["source_crop_bbox"][0] + endpoint_1000[0] * crop_width / 1000.0,
                    metadata["source_crop_bbox"][1] + endpoint_1000[1] * crop_height / 1000.0,
                )
                corner_midpoint_residual_px = round(math.dist(corner_midpoint, endpoint_source), 3)
            semantic_row.update(
                {
                    "candidate_id": candidate["candidate_id"],
                    "visible_side_horizontal": "right" if math.cos(angle) >= 0 else "left",
                    "visible_side_depth": (
                        "nearer"
                        if math.sin(angle) > 0.2
                        else "farther"
                        if math.sin(angle) < -0.2
                        else "same_depth"
                    ),
                    "building_ground_center_source_px": [
                        round(
                            metadata["source_crop_bbox"][0] + center_1000[0] * crop_width / 1000.0,
                            2,
                        ),
                        round(
                            metadata["source_crop_bbox"][1] + center_1000[1] * crop_height / 1000.0,
                            2,
                        ),
                    ],
                    "visible_side_ground_midpoint_source_px": [
                        round(
                            metadata["source_crop_bbox"][0]
                            + endpoint_1000[0] * crop_width / 1000.0,
                            2,
                        ),
                        round(
                            metadata["source_crop_bbox"][1]
                            + endpoint_1000[1] * crop_height / 1000.0,
                            2,
                        ),
                    ],
                    "visible_short_end_ground_corners_source_px": end_corners_source_px,
                    "short_end_corners_confidence": normalized_confidence(
                        semantic_row.get("short_end_corners_confidence"), 0.0
                    ),
                    "short_end_corner_midpoint_residual_px": corner_midpoint_residual_px,
                    "raw_camera_rectified_angle_deg": round(top_angle, 6),
                    "orientation_candidate_score": round(candidate_score, 6),
                    "orientation_line_support": candidate["line_support"],
                    "grounded_alignment_deg": round(grounded_alignment_deg, 6),
                }
            )
            rows.append(semantic_row)
        except (TypeError, ValueError, KeyError) as exc:
            print(f"Invalid grounded orientation for {building_id}: {exc}")
            print(f"Raw semantic response for {building_id}:\n{semantic_response}")
            raise
        raw_responses.append({"id": building_id, "semantic_response": semantic_response})
    try:
        orientations = normalize_rows(rows, expected_ids, candidates_by_building)
    except ValueError:
        print("Raw orientation responses before validation failure:")
        print(json.dumps(raw_responses, indent=2))
        raise

    json_path = args.output_dir / f"{args.image.stem}_building_orientations.json"
    overlay_path = args.output_dir / f"{args.image.stem}_building_orientations.png"
    document = {
        "stage": "automatic_building_orientation_inference",
        "image": str(args.image.resolve()),
        "image_size": list(image.size),
        "image_sha256": file_sha256(args.image),
        "detection_source": str(args.detection_json.resolve()),
        "detection_sha256": file_hash(args.detection_json),
        "model": args.model,
        "method": "second_stage_vlm_grounded_visible_end_corners_plus_mask_structural_axis_fusion",
        "orientation_definition": {
            "axis": "building_long_ground_axis_in_rectified_orthographic_top_view",
            "vlm_output": "grounded_center_visible_short_end_midpoint_and_ground_corners",
            "camera_rectification": {
                "projection": "orthographic",
                "pitch_deg": args.camera_pitch_deg,
                "azimuth_deg": args.camera_azimuth_deg,
                "roll_deg": 0,
            },
            "top_view_mapping": {"painting_right": "right", "painting_far_upper": "up"},
            "clock_reference": {"12": "up", "3": "right", "6": "down", "9": "left"},
            "directed_endpoint": "visible_left_or_right_short_end",
            "range": "full_12_hour_clock",
            "quantization_minutes": 30,
        },
        "semantic_prompt_template": SEMANTIC_ORIENTATION_PROMPT,
        "fusion_policy": {
            "candidate_vocabulary": "24 directed half-hour ground-plane axes",
            "hard_constraints": "same left/right and nearer/farther quadrant as VLM points",
            "score": "normalized_mask_hough_support + 1.1 * cosine(point_angle_difference)",
            "scene_specific_labels_or_coordinates": False,
        },
        "candidate_mapping": [
            {"id": building_id, "candidates": candidates_by_building[building_id]}
            for building_id in expected_ids
        ],
        "candidate_sheets": candidate_sheet_paths,
        "semantic_analysis_sheets": semantic_sheet_paths,
        "raw_responses": raw_responses,
        "grounding_reused_from": (
            str(args.reuse_grounding_json.resolve()) if args.reuse_grounding_json else None
        ),
        "orientations": orientations,
    }
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    draw_orientation_overlay(image, instances, orientations).save(overlay_path)
    print(
        json.dumps(
            {"count": len(orientations), "json": str(json_path), "overlay": str(overlay_path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
