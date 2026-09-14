#!/usr/bin/env python3
"""Detect whole building instances in Chinese paintings.

The pipeline is intentionally conservative:

1. a building-only YOLO detector proposes boxes on the full image and on
   overlapping crops;
2. SAM 2.1 converts the boxes into visible-region masks;
3. a separately trained crop classifier rejects mountains, vegetation,
   fences, and other building-like false positives;
4. mask containment consolidates boxes that duplicate a stronger candidate;
5. generic mask-shape checks reject fragmented roof pieces and shallow awnings
   that do not contain enough of a whole building, without hard-coding a count.

The JSON and overlay are the automatic whole-house parsing gate used before
face decomposition and cuboid fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = Path("models/building_detector.pt")
DEFAULT_CLASSIFIER = Path("models/building_classifier.pt")
DEFAULT_SAM = "facebook/sam2.1-hiera-small"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_proposal_cache(path: Path, image_path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = document.get("image_sha256")
    actual_hash = file_sha256(image_path)
    if expected_hash and expected_hash != actual_hash:
        raise SystemExit(
            f"Proposal cache does not belong to this image: expected {expected_hash}, got {actual_hash}"
        )
    proposals = document.get("proposals", document.get("detections"))
    if not isinstance(proposals, list):
        raise SystemExit(f"Invalid proposal cache: {path}")
    normalized = []
    for proposal in proposals:
        record = dict(proposal)
        if "detector_score" not in record and "score" in record:
            record["detector_score"] = record.pop("score")
        record.setdefault("proposal_source", "cached_automatic_detector")
        if "bbox" in record and "detector_score" in record:
            normalized.append(record)
    return normalized


def save_proposal_cache(
    path: Path,
    image_path: Path,
    detector: Path,
    proposals: list[dict],
    detector_confidence: float,
    box_nms_iou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "image": str(image_path.resolve()),
        "image_sha256": file_sha256(image_path),
        "generator": {
            "detector": str(detector.resolve()),
            "method": "full-image plus overlapping-tile inference, geometric filtering, box NMS",
            "detector_confidence": detector_confidence,
            "box_nms_iou": box_nms_iou,
            "note": "Automatic model output cache; no boxes were manually drawn or edited.",
        },
        "proposals": proposals,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def box_nms(items: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(items, key=lambda value: value["detector_score"], reverse=True):
        if all(box_iou(item["bbox"], previous["bbox"]) < threshold for previous in kept):
            kept.append(item)
    return kept


def add_whole_building_context_variants(
    items: list[dict], image_width: int, image_height: int
) -> list[dict]:
    """Add generic context boxes that can recover a roof and its walls.

    Object detectors often lock onto the most distinctive horizontal eave and
    truncate the roof above or walls below it.  These variants give the later
    whole-building classifier alternatives with architectural context.  The
    transforms depend only on each proposal's own size, never scene position or
    an expected instance count.
    """
    transforms = (
        ("original", 0.0, 0.0, 0.0, 0.0),
        ("roof_wall_recovery", 0.08, 0.80, 0.08, 0.12),
        ("balanced_context", 0.18, 0.45, 0.18, 0.28),
    )
    variants: list[dict] = []
    for family, item in enumerate(items):
        x1, y1, x2, y2 = map(float, item["bbox"])
        width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        for name, left, up, right, down in transforms:
            box = [
                max(0.0, x1 - left * width),
                max(0.0, y1 - up * height),
                min(float(image_width), x2 + right * width),
                min(float(image_height), y2 + down * height),
            ]
            record = dict(item)
            record["bbox"] = [round(value, 2) for value in box]
            record["proposal_family"] = family
            record["proposal_variant"] = name
            record["base_proposal_bbox"] = [round(value, 2) for value in (x1, y1, x2, y2)]
            variants.append(record)
    return variants


def spaced_positions(start: int, stop: int, window: int, stride: int) -> list[int]:
    """Cover an interval while avoiding a nearly duplicated final crop."""
    if window >= stop - start:
        return [start]
    positions = []
    value = start
    while value + window + stride <= stop:
        positions.append(value)
        value += stride
    final = stop - window
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def crop_with_context(image: Image.Image, box: list[float], expansion: float = 0.0) -> Image.Image:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return image.crop(
        (
            max(0, math.floor(x1 - width * expansion)),
            max(0, math.floor(y1 - height * expansion)),
            min(image.width, math.ceil(x2 + width * expansion)),
            min(image.height, math.ceil(y2 + height * expansion)),
        )
    ).convert("RGB")


def propose_boxes(
    image: Image.Image,
    model: YOLO,
    device: str,
    confidence: float,
    image_size: int,
) -> list[dict]:
    width, height = image.size
    # The band is a painting prior, not a scene-specific coordinate: in the
    # supplied Chinese landscapes, buildings occupy the middle/lower ground.
    # Full-image inference still permits a strong building elsewhere.
    tile_width = min(width, max(256, round(width * 0.43)))
    tile_height = min(height, max(256, round(height * 0.50)))
    x_positions = spaced_positions(0, width, tile_width, max(64, round(width * 0.185)))
    band_start = max(0, round(height * 0.30))
    band_stop = min(height, round(height * 0.88))
    y_positions = spaced_positions(
        band_start, band_stop, tile_height, max(48, round(height * 0.08))
    )

    views: list[tuple[Image.Image, int, int, str]] = [(image, 0, 0, "full")]
    for row, y0 in enumerate(y_positions):
        for column, x0 in enumerate(x_positions):
            views.append(
                (
                    image.crop((x0, y0, x0 + tile_width, y0 + tile_height)),
                    x0,
                    y0,
                    f"tile_r{row:02d}_c{column:02d}",
                )
            )

    proposals: list[dict] = []
    for view, offset_x, offset_y, source in views:
        result = model.predict(
            source=np.asarray(view),
            conf=confidence,
            iou=0.70,
            imgsz=image_size,
            device=device,
            half=True,
            max_det=80,
            verbose=False,
        )[0]
        for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = box
            global_box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
            box_width = max(0.0, global_box[2] - global_box[0])
            box_height = max(0.0, global_box[3] - global_box[1])
            area_ratio = box_width * box_height / max(width * height, 1)
            # Reject tiny brush marks and whole-landscape proposals. These are
            # generic scale/aspect checks and do not encode any expected count.
            if not (0.0015 <= area_ratio <= 0.14):
                continue
            if box_width > width * 0.48 or box_height > height * 0.38:
                continue
            proposals.append(
                {
                    "bbox": [round(float(value), 2) for value in global_box],
                    "detector_score": round(float(score), 6),
                    "proposal_source": source,
                }
            )
    return proposals


def classify_segmented_regions(
    image: Image.Image,
    items: list[dict],
    masks: list[np.ndarray],
    model: YOLO,
    device: str,
    threshold: float,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    if not items:
        return [], [], []
    # Keep these as PIL/RGB images. Converting them to NumPy here routes the
    # Ultralytics classifier through its OpenCV/BGR input path and materially
    # changes predictions on sepia paintings.
    crops = [crop_with_context(image, item["bbox"]) for item in items]
    results = model.predict(source=crops, imgsz=224, device=device, verbose=False)
    accepted, accepted_masks, rejected = [], [], []
    for item, mask, result in zip(items, masks, results):
        score = float(result.probs.data[0].cpu())
        item = dict(item)
        item["classifier_building_score"] = round(score, 6)
        item["classifier_decision"] = "building" if score >= threshold else "not_building"
        if score >= threshold:
            accepted.append(item)
            accepted_masks.append(mask)
        else:
            rejected.append(item)
    return accepted, accepted_masks, rejected


def segment_boxes(
    image: Image.Image,
    items: list[dict],
    model_name: str,
    device: str,
    confidence_threshold: float,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    if not items:
        return items, [], []
    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name).to(device)
    model.eval()
    boxes = [item["bbox"] for item in items]
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
    model_inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor) and key != "original_sizes"
        else value
        for key, value in inputs.items()
        if key != "original_sizes"
    }
    with torch.inference_mode():
        outputs = model(**model_inputs, multimask_output=True)
    candidates = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
        max_hole_area=64,
        max_sprinkle_area=32,
    )[0]
    qualities = outputs.iou_scores.detach().cpu()[0]

    masks: list[np.ndarray] = []
    segmented: list[dict] = []
    rejected: list[dict] = []
    image_area = image.width * image.height
    for index, item in enumerate(items):
        best = int(torch.argmax(qualities[index]).item())
        mask = candidates[index, best].numpy().astype(bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        record = dict(item)
        record["sam_predicted_iou"] = round(float(qualities[index, best]), 6)
        record["mask_area_pixels"] = int(mask.sum())
        record["mask_area_ratio"] = round(float(mask.sum() / image_area), 8)
        record["mask_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        if record["sam_predicted_iou"] < confidence_threshold:
            record["rejection_reason"] = "low_sam_mask_quality"
            rejected.append(record)
            continue
        segmented.append(record)
        masks.append(mask)
    return segmented, masks, rejected


def filter_whole_building_structure(
    items: list[dict],
    masks: list[np.ndarray],
    small_fragment_area: float,
    small_fragment_fill: float,
    awning_aspect: float,
    awning_fill: float,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    """Reject roof fragments and shallow canopies without assuming a count.

    A whole painted building can be small or wide, but tiny disconnected masks
    with little support inside their envelope are normally roof fragments. A
    very wide, low-fill envelope is likewise characteristic of an eave, awning,
    or covered walk rather than a complete building body.
    """
    accepted: list[dict] = []
    accepted_masks: list[np.ndarray] = []
    rejected: list[dict] = []
    for item, mask in zip(items, masks):
        record = dict(item)
        x1, y1, x2, y2 = record["mask_bbox"]
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        fill_ratio = float(record["mask_area_pixels"]) / float(width * height)
        aspect_ratio = width / height
        record["mask_fill_ratio"] = round(fill_ratio, 6)
        record["mask_aspect_ratio"] = round(aspect_ratio, 6)

        if record["mask_area_ratio"] < small_fragment_area and fill_ratio < small_fragment_fill:
            record["rejection_reason"] = "small_fragment_without_whole_building_support"
            rejected.append(record)
        elif aspect_ratio > awning_aspect and fill_ratio < awning_fill:
            record["rejection_reason"] = "shallow_awning_or_canopy"
            rejected.append(record)
        else:
            accepted.append(record)
            accepted_masks.append(mask)
    # Deduplication establishes a stable depth/left-to-right order, but the
    # final structure gate may remove candidates in the middle of that order.
    # Renumber only after the final acceptance decision so IDs are contiguous.
    for number, record in enumerate(accepted, 1):
        record["id"] = f"House_{number:02d}"
    return accepted, accepted_masks, rejected


def deduplicate_masks(
    items: list[dict],
    masks: list[np.ndarray],
    containment_threshold: float,
    box_containment_threshold: float,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    """Keep the strongest instance when two masks substantially contain one another.

    Ranking by model agreement instead of raw area is important for compounds:
    a large low-quality mask may join two adjacent houses, while a tighter box
    gives a better detector/classifier/SAM consensus for one actual building.
    """

    def quality(index: int) -> float:
        item = items[index]
        return (
            float(item["detector_score"])
            * float(item["classifier_building_score"])
            * max(float(item["sam_predicted_iou"]), 0.05)
        )

    order = sorted(
        range(len(items)),
        key=lambda index: (quality(index), items[index]["mask_area_pixels"]),
        reverse=True,
    )
    kept: list[int] = []
    rejected: list[dict] = []
    for index in order:
        area = max(int(masks[index].sum()), 1)
        duplicate_of = None
        duplicate_containment = 0.0
        for previous in kept:
            intersection = int(np.logical_and(masks[index], masks[previous]).sum())
            previous_area = max(int(masks[previous].sum()), 1)
            mask_containment = intersection / min(area, previous_area)
            current_box, previous_box = items[index]["bbox"], items[previous]["bbox"]
            box_intersection = max(
                0.0, min(current_box[2], previous_box[2]) - max(current_box[0], previous_box[0])
            ) * max(
                0.0, min(current_box[3], previous_box[3]) - max(current_box[1], previous_box[1])
            )
            current_box_area = max(
                1.0, (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            )
            previous_box_area = max(
                1.0, (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
            )
            box_containment = box_intersection / min(current_box_area, previous_box_area)
            if (
                mask_containment >= containment_threshold
                or box_containment >= box_containment_threshold
            ):
                duplicate_of = previous
                duplicate_containment = max(mask_containment, box_containment)
                break
        if duplicate_of is None:
            kept.append(index)
        else:
            record = dict(items[index])
            record["rejection_reason"] = "mask_duplicate_of_higher_quality_building"
            record["contained_fraction"] = round(duplicate_containment, 6)
            record["duplicate_of_candidate_bbox"] = items[duplicate_of]["bbox"]
            record["model_agreement_score"] = round(quality(index), 8)
            rejected.append(record)

    # Stable IDs follow painting depth (upper first), then left-to-right.
    kept.sort(key=lambda index: (items[index]["mask_bbox"][1], items[index]["mask_bbox"][0]))
    final_items, final_masks = [], []
    for number, index in enumerate(kept, 1):
        record = dict(items[index])
        record["id"] = f"House_{number:02d}"
        record["model_agreement_score"] = round(quality(index), 8)
        final_items.append(record)
        final_masks.append(masks[index])
    return final_items, final_masks, rejected


def split_vlm_merged_parents(
    items: list[dict],
    masks: list[np.ndarray],
    containment_threshold: float,
    box_containment_threshold: float,
    child_containment: float,
    child_confidence: float,
    child_max_iou: float,
    child_min_separation: float,
    child_max_area_fraction: float,
    child_combined_area: float,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    """Suppress a VLM group box when two distinct detector children support it.

    Multimodal grounding is good at semantic completeness but can merge an
    adjacent compound into one box.  The trained local detector is better at
    proposing individual roofs.  A parent is split only when two independently
    strong, mutually distinct child instances lie mostly inside it.  All
    thresholds are relative to the candidate boxes; no scene coordinates or
    expected counts are used.
    """
    has_vlm_parent = any(
        str(item.get("proposal_source", "")).startswith("qwen3_vl_whole_building_grounding")
        for item in items
    )
    if not has_vlm_parent:
        return items, masks, []

    def child_key(item: dict) -> tuple:
        return (
            str(item.get("proposal_source", "")),
            tuple(round(float(value), 2) for value in item["bbox"]),
        )

    local_pairs = [
        (item, mask)
        for item, mask in zip(items, masks)
        if not str(item.get("proposal_source", "")).startswith("qwen3_vl_whole_building_grounding")
        and float(item.get("classifier_building_score", 0.0)) >= child_confidence
    ]
    unique_children: list[dict] = []
    if local_pairs:
        local_items, local_masks = map(list, zip(*local_pairs))
        unique_children, _, _ = deduplicate_masks(
            local_items,
            local_masks,
            containment_threshold,
            box_containment_threshold,
        )

    rejected_indices: set[int] = set()
    rejected: list[dict] = []
    selected_child_keys: set[tuple] = set()
    for parent_index, parent in enumerate(items):
        if not str(parent.get("proposal_source", "")).startswith(
            "qwen3_vl_whole_building_grounding"
        ):
            continue
        px1, py1, px2, py2 = map(float, parent["bbox"])
        parent_width, parent_height = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
        parent_area = parent_width * parent_height
        eligible: list[dict] = []
        for child in unique_children:
            cx1, cy1, cx2, cy2 = map(float, child["bbox"])
            child_area = max((cx2 - cx1) * (cy2 - cy1), 1.0)
            intersection = max(0.0, min(px2, cx2) - max(px1, cx1)) * max(
                0.0, min(py2, cy2) - max(py1, cy1)
            )
            if child_area / parent_area > child_max_area_fraction:
                continue
            if intersection / child_area >= child_containment:
                eligible.append(child)

        supporting_pair = None
        for first_index, first in enumerate(eligible):
            for second in eligible[first_index + 1 :]:
                first_box, second_box = first["bbox"], second["bbox"]
                first_area = max((first_box[2] - first_box[0]) * (first_box[3] - first_box[1]), 1.0)
                second_area = max(
                    (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]), 1.0
                )
                first_center = (
                    (first_box[0] + first_box[2]) / 2,
                    (first_box[1] + first_box[3]) / 2,
                )
                second_center = (
                    (second_box[0] + second_box[2]) / 2,
                    (second_box[1] + second_box[3]) / 2,
                )
                separation = max(
                    abs(first_center[0] - second_center[0]) / parent_width,
                    abs(first_center[1] - second_center[1]) / parent_height,
                )
                if (
                    box_iou(first_box, second_box) <= child_max_iou
                    and separation >= child_min_separation
                    and (first_area + second_area) / parent_area >= child_combined_area
                ):
                    supporting_pair = (first, second)
                    break
            if supporting_pair:
                break
        if supporting_pair:
            record = dict(parent)
            record["rejection_reason"] = "vlm_group_box_split_by_distinct_complete_children"
            record["supporting_child_bboxes"] = [child["bbox"] for child in supporting_pair]
            rejected.append(record)
            rejected_indices.add(parent_index)
            selected_child_keys.update(child_key(child) for child in supporting_pair)
            for other_index, other in enumerate(items):
                if other_index == parent_index or not str(
                    other.get("proposal_source", "")
                ).startswith("qwen3_vl_whole_building_grounding"):
                    continue
                ox1, oy1, ox2, oy2 = map(float, other["bbox"])
                other_area = max((ox2 - ox1) * (oy2 - oy1), 1.0)
                intersection = max(0.0, min(px2, ox2) - max(px1, ox1)) * max(
                    0.0, min(py2, oy2) - max(py1, oy1)
                )
                if (
                    other_area / parent_area <= child_max_area_fraction
                    and intersection / other_area >= child_containment
                ):
                    other_record = dict(other)
                    other_record["rejection_reason"] = (
                        "redundant_vlm_subbox_after_parent_decomposition"
                    )
                    rejected.append(other_record)
                    rejected_indices.add(other_index)

    # In the ensemble, VLM boxes own semantic recall and local-detector boxes
    # are decomposition evidence only.  Let through only the exact child pair
    # that justified splitting a merged VLM parent; otherwise low-threshold
    # local roof fragments would become independent instances again.
    for index, item in enumerate(items):
        if str(item.get("proposal_source", "")).startswith("qwen3_vl_whole_building_grounding"):
            continue
        if child_key(item) not in selected_child_keys:
            record = dict(item)
            record["rejection_reason"] = "local_proposal_not_needed_for_vlm_parent_decomposition"
            rejected.append(record)
            rejected_indices.add(index)

    kept_items = [item for index, item in enumerate(items) if index not in rejected_indices]
    kept_masks = [mask for index, mask in enumerate(masks) if index not in rejected_indices]
    return kept_items, kept_masks, rejected


def mask_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 24:
            continue
        epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons[:12]


def save_outputs(
    image: Image.Image,
    image_path: Path,
    output_dir: Path,
    items: list[dict],
    masks: list[np.ndarray],
    rejected_sam: list[dict],
    rejected_structure: list[dict],
    rejected_classifier: list[dict],
    rejected_merged_parents: list[dict],
    rejected_duplicates: list[dict],
    settings: dict,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_building_instances.png"
    json_path = output_dir / f"{stem}_building_instances.json"
    mask_path = output_dir / f"{stem}_building_instance_mask.png"

    label_map = np.zeros((image.height, image.width), dtype=np.uint8)
    colors = [
        (230, 57, 70),
        (29, 142, 168),
        (244, 162, 97),
        (42, 157, 143),
        (131, 56, 236),
        (233, 196, 106),
        (76, 110, 245),
        (231, 111, 81),
    ]
    tint = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for index, mask in enumerate(masks, 1):
        label_map[mask] = index
        color = colors[(index - 1) % len(colors)]
        tint[mask] = (*color, 92)
    overlay = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(tint, "RGBA"))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    for index, item in enumerate(items):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = item["mask_bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=4)
        label = f"{item['id']}  det={item['detector_score']:.2f} cls={item['classifier_building_score']:.2f}"
        text_box = draw.textbbox((x1 + 3, y1 + 3), label, font=font)
        draw.rectangle(
            (text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2),
            fill=(18, 18, 18, 225),
        )
        draw.text((x1 + 3, y1 + 3), label, fill=(255, 255, 255, 255), font=font)
        item["visible_region_polygons"] = mask_polygons(masks[index])
    overlay.convert("RGB").save(overlay_path)
    Image.fromarray(label_map, mode="L").save(mask_path)

    document = {
        "image": str(image_path.resolve()),
        "image_size": [image.width, image.height],
        "stage": "automatic_whole_building_instance_parsing",
        "models": {
            "proposal_detector": str(settings["detector"]),
            "semantic_proposal_models": settings.get("semantic_proposal_models", []),
            "proposal_classifier": str(settings["classifier"]),
            "visible_region_segmenter": settings["sam_model"],
        },
        "settings": settings,
        "count": len(items),
        "instances": items,
        "rejected": {
            "sam_quality": rejected_sam,
            "whole_building_structure": rejected_structure,
            "classifier": rejected_classifier,
            "merged_vlm_parent": rejected_merged_parents,
            "mask_containment": rejected_duplicates,
        },
    }
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    return json_path, overlay_path, mask_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/house_detection"))
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument(
        "--proposals-json",
        type=Path,
        action="append",
        help="reuse a hash-checked cache of automatically generated detector proposals",
    )
    parser.add_argument(
        "--save-proposals-json", type=Path, help="save generated local-detector proposals"
    )
    parser.add_argument(
        "--proposals-only", action="store_true", help="stop after saving local-detector proposals"
    )
    parser.add_argument("--sam-model", default=DEFAULT_SAM)
    parser.add_argument("--device", required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.03)
    parser.add_argument("--classifier-confidence", type=float, default=0.85)
    parser.add_argument("--box-nms", type=float, default=0.25)
    parser.add_argument("--mask-containment", type=float, default=0.60)
    parser.add_argument("--box-containment", type=float, default=0.68)
    parser.add_argument("--sam-confidence", type=float, default=0.40)
    parser.add_argument("--small-fragment-area", type=float, default=0.006)
    parser.add_argument("--small-fragment-fill", type=float, default=0.35)
    parser.add_argument("--awning-aspect", type=float, default=2.70)
    parser.add_argument("--awning-fill", type=float, default=0.50)
    parser.add_argument("--parent-child-containment", type=float, default=0.74)
    parser.add_argument("--split-child-confidence", type=float, default=0.85)
    parser.add_argument("--split-child-max-iou", type=float, default=0.32)
    parser.add_argument("--split-child-min-separation", type=float, default=0.18)
    parser.add_argument("--split-child-max-area", type=float, default=0.82)
    parser.add_argument("--split-child-combined-area", type=float, default=0.65)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument(
        "--no-context-variants",
        action="store_true",
        help="disable scale-relative proposal variants that recover roof/wall context",
    )
    args = parser.parse_args()

    if not args.detector.is_file() or not args.classifier.is_file():
        raise SystemExit("Trained detector/classifier weights are missing; see README.md")
    image = Image.open(args.image).convert("RGB")

    if args.proposals_json:
        proposals = []
        semantic_proposal_models = []
        for proposal_cache in args.proposals_json:
            cache_document = json.loads(proposal_cache.read_text(encoding="utf-8"))
            if cache_document.get("model"):
                semantic_proposal_models.append(str(cache_document["model"]))
            proposals.extend(load_proposal_cache(proposal_cache, args.image))
    else:
        semantic_proposal_models = []
        detector = YOLO(str(args.detector))
        proposals = propose_boxes(
            image, detector, args.device, args.detector_confidence, args.image_size
        )
        proposals = box_nms(proposals, args.box_nms)
        if args.save_proposals_json:
            save_proposal_cache(
                args.save_proposals_json,
                args.image,
                args.detector,
                proposals,
                args.detector_confidence,
                args.box_nms,
            )
        del detector
        torch.cuda.empty_cache()
        if args.proposals_only:
            print(json.dumps({"count": len(proposals), "json": str(args.save_proposals_json)}))
            return 0

    base_proposal_count = len(proposals)
    if not args.no_context_variants:
        proposals = add_whole_building_context_variants(proposals, image.width, image.height)

    segmented, masks, rejected_sam = segment_boxes(
        image, proposals, args.sam_model, args.device, args.sam_confidence
    )
    torch.cuda.empty_cache()
    classifier = YOLO(str(args.classifier))
    classified, masks, rejected_classifier = classify_segmented_regions(
        image, segmented, masks, classifier, args.device, args.classifier_confidence
    )
    del classifier
    torch.cuda.empty_cache()

    classified, masks, rejected_merged_parents = split_vlm_merged_parents(
        classified,
        masks,
        args.mask_containment,
        args.box_containment,
        args.parent_child_containment,
        args.split_child_confidence,
        args.split_child_max_iou,
        args.split_child_min_separation,
        args.split_child_max_area,
        args.split_child_combined_area,
    )
    items, masks, rejected_duplicates = deduplicate_masks(
        classified, masks, args.mask_containment, args.box_containment
    )
    items, masks, rejected_structure = filter_whole_building_structure(
        items,
        masks,
        args.small_fragment_area,
        args.small_fragment_fill,
        args.awning_aspect,
        args.awning_fill,
    )
    settings = {
        "detector": str(args.detector.resolve()),
        "classifier": str(args.classifier.resolve()),
        "sam_model": args.sam_model,
        "semantic_proposal_models": sorted(set(semantic_proposal_models)),
        "device": args.device,
        "detector_confidence": args.detector_confidence,
        "classifier_confidence": args.classifier_confidence,
        "box_nms_iou": args.box_nms,
        "mask_containment": args.mask_containment,
        "box_containment": args.box_containment,
        "sam_confidence": args.sam_confidence,
        "small_fragment_area_ratio": args.small_fragment_area,
        "small_fragment_fill_ratio": args.small_fragment_fill,
        "awning_aspect_ratio": args.awning_aspect,
        "awning_fill_ratio": args.awning_fill,
        "parent_child_containment": args.parent_child_containment,
        "split_child_confidence": args.split_child_confidence,
        "split_child_max_iou": args.split_child_max_iou,
        "split_child_min_separation": args.split_child_min_separation,
        "split_child_max_area_fraction": args.split_child_max_area,
        "split_child_combined_area": args.split_child_combined_area,
        "image_size": args.image_size,
        "raw_proposals": base_proposal_count,
        "proposals_with_context_variants": len(proposals),
        "whole_building_context_variants": not args.no_context_variants,
        "proposal_caches": [str(path.resolve()) for path in args.proposals_json]
        if args.proposals_json
        else [],
    }
    json_path, overlay_path, mask_path = save_outputs(
        image,
        args.image,
        args.output_dir,
        items,
        masks,
        rejected_sam,
        rejected_structure,
        rejected_classifier,
        rejected_merged_parents,
        rejected_duplicates,
        settings,
    )
    print(
        json.dumps(
            {
                "count": len(items),
                "base_proposals_after_box_nms": base_proposal_count,
                "proposals_with_context_variants": len(proposals),
                "sam_rejected": len(rejected_sam),
                "structure_rejected": len(rejected_structure),
                "classifier_rejected": len(rejected_classifier),
                "merged_parent_rejected": len(rejected_merged_parents),
                "containment_rejected": len(rejected_duplicates),
                "json": str(json_path),
                "overlay": str(overlay_path),
                "mask": str(mask_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
