#!/usr/bin/env python3
"""CPU cuboid replay from cached detections, directions and explicit scene priors.

This is a research reconstruction baseline, not a general painting detector.
Camera and terrain are provided by the caller. The preliminary semantic panel
is constructed from cuboid assumptions, not independently segmented faces.
No local model, GPU, network request or mesh framework is used by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def hex_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


PALETTE_HEX = json.loads((ROOT / "contracts" / "palette.json").read_text())
PALETTE = {name: hex_rgb(value) for name, value in PALETTE_HEX.items()}
FACE_COLOR = {
    "front": PALETTE["house_front"],
    "back": PALETTE["house_back"],
    "left": PALETTE["house_left"],
    "right": PALETTE["house_right"],
    "top": PALETTE["house_top"],
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def px_polygon(poly: Sequence[Point2], width: int, height: int) -> List[Point2]:
    return [(x * width, y * height) for x, y in poly]


def round_poly(poly: Sequence[Point2]) -> List[List[float]]:
    return [[round(x, 2), round(y, 2)] for x, y in poly]


def polygon_centroid(poly: Sequence[Point2]) -> Point2:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def shade(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(clamp(c * factor, 0, 255)) for c in color)


@dataclass
class Camera:
    width: int
    height: int
    pitch_deg: float
    azimuth_deg: float
    scale: float
    offset_y_ratio: float

    @property
    def pitch(self) -> float:
        return math.radians(self.pitch_deg)

    @property
    def azimuth(self) -> float:
        return math.radians(self.azimuth_deg)

    @property
    def offset_x(self) -> float:
        return self.width / 2.0

    @property
    def offset_y(self) -> float:
        return self.height * self.offset_y_ratio

    def project(self, p: Point3) -> Point2:
        x, y, z = p
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        screen_h = ca * x - sa * y
        far_h = sa * x + ca * y
        sx = self.offset_x + self.scale * screen_h
        sy = self.offset_y - self.scale * (math.sin(self.pitch) * far_h + math.cos(self.pitch) * z)
        return sx, sy

    def depth(self, p: Point3) -> float:
        x, y, z = p
        far_h = math.sin(self.azimuth) * x + math.cos(self.azimuth) * y
        return math.cos(self.pitch) * far_h - math.sin(self.pitch) * z

    def ground_from_pixel(self, p: Point2) -> Point2:
        sx, sy = p
        screen_h = (sx - self.offset_x) / self.scale
        far_h = (self.offset_y - sy) / (self.scale * math.sin(self.pitch))
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        x = ca * screen_h + sa * far_h
        y = -sa * screen_h + ca * far_h
        return x, y

    def projected_ground_axis(self, yaw_deg: float) -> Point2:
        """Unit image vector for a world-ground direction under this camera."""
        relative_yaw = math.radians(yaw_deg - self.azimuth_deg)
        dx = math.cos(relative_yaw)
        dy = -math.sin(self.pitch) * math.sin(relative_yaw)
        magnitude = max(math.hypot(dx, dy), 1e-9)
        return dx / magnitude, dy / magnitude


@dataclass
class HouseObservation:
    """A whole-house parse expressed only in source-image measurements."""

    id: str
    ground_center_px: Point2
    broad_span_px: float
    top_recede_px: float
    wall_height_px: float
    yaw_deg: float
    broad_face: str
    dimension_evidence: dict = field(default_factory=dict)

    def visible_faces(self) -> List[str]:
        return [self.broad_face, "left" if self.yaw_deg > 0 else "right", "top"]

    def face_polygons(self, camera: Camera) -> Dict[str, List[Point2]]:
        """Construct raw 2D face quads without creating any 3D geometry."""
        long_unit = camera.projected_ground_axis(self.yaw_deg)
        depth_unit = camera.projected_ground_axis(self.yaw_deg + 90.0)
        # Keep the signed local depth axis identical to House.axes().  The
        # front/back face definitions below already choose +/- depth; flipping
        # the axis here would translate every raw face by one full depth span
        # for half of all possible yaws.
        long_vec = (self.broad_span_px * long_unit[0], self.broad_span_px * long_unit[1])
        depth_vec = (self.top_recede_px * depth_unit[0], self.top_recede_px * depth_unit[1])
        vertical_vec = (0.0, -self.wall_height_px)
        cx, cy = self.ground_center_px

        def p(along: float, across: float, top: float) -> Point2:
            return (
                cx + along * long_vec[0] + across * depth_vec[0] + top * vertical_vec[0],
                cy + along * long_vec[1] + across * depth_vec[1] + top * vertical_vec[1],
            )

        faces = {
            "front": [p(-0.5, -0.5, 0), p(0.5, -0.5, 0), p(0.5, -0.5, 1), p(-0.5, -0.5, 1)],
            "back": [p(0.5, 0.5, 0), p(-0.5, 0.5, 0), p(-0.5, 0.5, 1), p(0.5, 0.5, 1)],
            "left": [p(-0.5, 0.5, 0), p(-0.5, -0.5, 0), p(-0.5, -0.5, 1), p(-0.5, 0.5, 1)],
            "right": [p(0.5, -0.5, 0), p(0.5, 0.5, 0), p(0.5, 0.5, 1), p(0.5, -0.5, 1)],
            "top": [p(-0.5, -0.5, 1), p(0.5, -0.5, 1), p(0.5, 0.5, 1), p(-0.5, 0.5, 1)],
        }
        return {name: faces[name] for name in self.visible_faces()}


@dataclass
class House:
    id: str
    center: Point3
    length: float
    depth: float
    height: float
    yaw_deg: float
    broad_face: str
    source_center_px: Point2

    @property
    def yaw(self) -> float:
        return math.radians(self.yaw_deg)

    def axes(self) -> Tuple[Point2, Point2]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c, s), (-s, c)

    def point(self, along: float, across: float, z: float) -> Point3:
        long_axis, depth_axis = self.axes()
        return (
            self.center[0] + along * long_axis[0] + across * depth_axis[0],
            self.center[1] + along * long_axis[1] + across * depth_axis[1],
            z,
        )

    def faces(self) -> Dict[str, List[Point3]]:
        half_length, d, h = self.length / 2.0, self.depth / 2.0, self.height
        return {
            "front": [
                self.point(-half_length, -d, 0),
                self.point(half_length, -d, 0),
                self.point(half_length, -d, h),
                self.point(-half_length, -d, h),
            ],
            "back": [
                self.point(half_length, d, 0),
                self.point(-half_length, d, 0),
                self.point(-half_length, d, h),
                self.point(half_length, d, h),
            ],
            "left": [
                self.point(-half_length, d, 0),
                self.point(-half_length, -d, 0),
                self.point(-half_length, -d, h),
                self.point(-half_length, d, h),
            ],
            "right": [
                self.point(half_length, -d, 0),
                self.point(half_length, d, 0),
                self.point(half_length, d, h),
                self.point(half_length, -d, h),
            ],
            "top": [
                self.point(-half_length, -d, h),
                self.point(half_length, -d, h),
                self.point(half_length, d, h),
                self.point(-half_length, d, h),
            ],
        }

    def visible_faces(self) -> List[str]:
        side = "left" if self.yaw_deg > 0 else "right"
        return [self.broad_face, side, "top"]

    def footprint(self) -> List[Point2]:
        half_length, d = self.length / 2.0, self.depth / 2.0
        return [
            self.point(-half_length, -d, 0)[:2],
            self.point(half_length, -d, 0)[:2],
            self.point(half_length, d, 0)[:2],
            self.point(-half_length, d, 0)[:2],
        ]


def contour_yaw_deg(instance: dict) -> float:
    """Estimate a conservative in-plane yaw from the largest SAM contour."""
    polygons = instance.get("visible_region_polygons", [])
    if not polygons:
        return 0.0
    polygon = max(polygons, key=len)
    if len(polygon) < 3:
        return 0.0
    cx = sum(point[0] for point in polygon) / len(polygon)
    cy = sum(point[1] for point in polygon) / len(polygon)
    sxx = sum((point[0] - cx) ** 2 for point in polygon)
    syy = sum((point[1] - cy) ** 2 for point in polygon)
    sxy = sum((point[0] - cx) * (point[1] - cy) for point in polygon)
    angle = math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy))
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return round(clamp(angle, -12.0, 12.0), 3)


def orientation_yaw_deg(row: dict) -> float:
    """Convert directed top-view clock angle to world yaw (+y is painting-far)."""
    clockwise_from_right = float(row["top_view_angle_deg_clockwise_from_right"])
    yaw = -clockwise_from_right
    while yaw <= -180.0:
        yaw += 360.0
    while yaw > 180.0:
        yaw -= 360.0
    return round(yaw, 3)


def inferred_mask_ground_y(instance: dict) -> float:
    """Ignore narrow SAM drips below a roof when inferring a hidden foot."""
    x1, y1, x2, y2 = map(float, instance["mask_bbox"])
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    broad_components = []
    for polygon in instance.get("visible_region_polygons", []):
        if not polygon:
            continue
        component_width = max(point[0] for point in polygon) - min(point[0] for point in polygon)
        if component_width >= width * 0.25:
            broad_components.append(max(point[1] for point in polygon))
    if not broad_components:
        return y2
    visible_bottom = float(max(broad_components))
    if visible_bottom < y2 - height * 0.25:
        # A roof-only component has an occluded wall below it. Complete only a
        # conservative fraction of the mask height; do not follow thin fence/
        # tree fragments all the way to the global mask bottom.
        visible_bottom += height * 0.15
    return min(y2, visible_bottom)


def packed_mask_area(mask: Image.Image) -> int:
    """Count set pixels in a Pillow mode-1 mask without NumPy."""
    return sum(byte.bit_count() for byte in mask.tobytes())


def mask_guided_length_depth_ratio(
    instance: dict,
    camera: Camera,
    ground_center_px: Point2,
    broad_span_px: float,
    wall_height_px: float,
    yaw_deg: float,
) -> dict:
    """Fit footprint ratio to the accepted SAM silhouette in image space.

    Direction is already fixed by the independent orientation pass.  This
    small discrete fit changes only the short footprint dimension. Nuisance
    image translations absorb an uncertain/occluded foot point but are never
    carried into scene placement. A 3:1 prior wins whenever the visible mask
    does not distinguish a different ratio clearly.
    """
    polygons = [poly for poly in instance.get("visible_region_polygons", []) if len(poly) >= 3]
    if not polygons:
        return {
            "ratio": 3.0,
            "accepted": False,
            "best_iou": 0.0,
            "fallback_iou": 0.0,
            "iou_gain_over_fallback": 0.0,
        }
    x1, y1, x2, y2 = map(float, instance["mask_bbox"])
    visible_width = max(12.0, x2 - x1)
    visible_height = max(12.0, y2 - y1)
    pad = max(32, round(max(visible_width, visible_height) * 0.18))
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(camera.width, int(math.ceil(x2 + pad)))
    bottom = min(camera.height, int(math.ceil(y2 + pad)))
    size = (max(1, right - left), max(1, bottom - top))
    target = Image.new("1", size, 0)
    target_draw = ImageDraw.Draw(target)
    for polygon in polygons:
        target_draw.polygon([(x - left, y - top) for x, y in polygon], fill=1)
    target_bytes = target.tobytes()
    target_area = packed_mask_area(target)
    if target_area <= 0:
        return {
            "ratio": 3.0,
            "accepted": False,
            "best_iou": 0.0,
            "fallback_iou": 0.0,
            "iou_gain_over_fallback": 0.0,
        }

    relative_yaw = math.radians(yaw_deg - camera.azimuth_deg)
    projected_long = camera.scale * math.hypot(
        math.cos(relative_yaw), math.sin(camera.pitch) * math.sin(relative_yaw)
    )
    projected_depth = camera.scale * math.hypot(
        math.sin(relative_yaw), math.sin(camera.pitch) * math.cos(relative_yaw)
    )
    depth_axis_horizontal_observability = abs(camera.projected_ground_axis(yaw_deg + 90.0)[0])
    inferred_length = broad_span_px / projected_long
    ratios = [1.5 + 0.25 * index for index in range(13)]  # 1.5 .. 4.5
    shift_step = max(8.0, min(30.0, max(visible_width, visible_height) * 0.06))
    shifts = [-shift_step, -shift_step / 2.0, 0.0, shift_step / 2.0, shift_step]
    scores: dict[float, float] = {}
    for ratio in ratios:
        projected_short_span = (inferred_length / ratio) * projected_depth
        best_iou = 0.0
        for shift_x in shifts:
            for shift_y in shifts:
                observation = HouseObservation(
                    str(instance["id"]),
                    (ground_center_px[0] + shift_x, ground_center_px[1] + shift_y),
                    broad_span_px,
                    projected_short_span,
                    wall_height_px,
                    yaw_deg,
                    "front",
                )
                candidate = Image.new("1", size, 0)
                candidate_draw = ImageDraw.Draw(candidate)
                for polygon in observation.face_polygons(camera).values():
                    candidate_draw.polygon(
                        [(x - left, y - top) for x, y in polygon],
                        fill=1,
                    )
                candidate_bytes = candidate.tobytes()
                candidate_area = packed_mask_area(candidate)
                intersection = sum(
                    (target_byte & candidate_byte).bit_count()
                    for target_byte, candidate_byte in zip(target_bytes, candidate_bytes)
                )
                union = target_area + candidate_area - intersection
                best_iou = max(best_iou, intersection / union if union else 0.0)
        scores[ratio] = best_iou
    best_ratio = max(ratios, key=lambda ratio: (scores[ratio], -abs(ratio - 3.0)))
    best_iou = scores[best_ratio]
    fallback_iou = scores[3.0]
    gain = best_iou - fallback_iou
    # When the projected short axis is nearly vertical, footprint depth and
    # wall/roof height are inseparable in one painting view. Likewise, a tiny
    # IoU improvement over 3:1 is not enough to override the stated fallback.
    accepted = depth_axis_horizontal_observability >= 0.35 and best_iou >= 0.38 and gain >= 0.05
    return {
        "ratio": best_ratio if accepted else 3.0,
        "accepted": accepted,
        "best_candidate_ratio": best_ratio,
        "best_iou": round(best_iou, 6),
        "fallback_iou": round(fallback_iou, 6),
        "iou_gain_over_fallback": round(gain, 6),
        "depth_axis_horizontal_observability": round(depth_axis_horizontal_observability, 6),
    }


def parse_house_observations(
    scene: dict,
    camera: Camera,
    automatic_detection: dict | None = None,
    automatic_orientation: dict | None = None,
) -> List[HouseObservation]:
    if automatic_detection is not None:
        if automatic_orientation is None:
            raise ValueError(
                "automatic building locations require the independent VLM orientation stage"
            )
        orientation_by_id = {row["id"]: row for row in automatic_orientation["orientations"]}
        detected_ids = {instance["id"] for instance in automatic_detection["instances"]}
        if set(orientation_by_id) != detected_ids:
            raise ValueError("orientation IDs do not exactly match accepted building instance IDs")
        observations = []
        for instance in automatic_detection["instances"]:
            orientation = orientation_by_id[instance["id"]]
            x1, y1, x2, y2 = map(float, instance["mask_bbox"])
            visible_width = max(12.0, x2 - x1)
            ground_y = inferred_mask_ground_y(instance)
            visible_height = max(12.0, ground_y - y1)
            # These measurements are deliberately image-space only. The
            # cuboid solver below is still the first place world dimensions
            # are created. Occluded buildings use the bottom of the visible
            # SAM region as their inferred ground-contact evidence.
            # SAM outlines include eaves, vegetation holes, and occasionally a
            # neighbouring wall. The upright cuboid body is therefore fitted
            # inside (not around) the full visible mask envelope.
            yaw_deg = orientation_yaw_deg(orientation)
            broad_span = max(camera.width * 0.08, visible_width * 0.78)
            relative_yaw = math.radians(yaw_deg - camera.azimuth_deg)
            projected_long = camera.scale * math.hypot(
                math.cos(relative_yaw), math.sin(camera.pitch) * math.sin(relative_yaw)
            )
            projected_depth = camera.scale * math.hypot(
                math.sin(relative_yaw), math.sin(camera.pitch) * math.cos(relative_yaw)
            )
            inferred_length = broad_span / projected_long
            mask_ratio_fit = mask_guided_length_depth_ratio(
                instance,
                camera,
                ((x1 + x2) / 2.0, min(camera.height - 1.0, ground_y)),
                broad_span,
                max(12.0, visible_height * 0.32),
                yaw_deg,
            )
            # The short wall's two grounded base corners are direct footprint
            # evidence. Project their image vector onto the known short axis;
            # vertical wall height and roof/eave overhang therefore do not leak
            # into depth as they did in the old visible-height heuristic.
            corner_pair = orientation.get("visible_short_end_ground_corners_source_px")
            corner_confidence = float(orientation.get("short_end_corners_confidence", 0.0))
            depth_unit = camera.projected_ground_axis(yaw_deg + 90.0)
            corner_span = 0.0
            corner_alignment = 0.0
            detected_ratio = None
            depth_source = "fallback_3_to_1_unresolved_short_end"
            if isinstance(corner_pair, list) and len(corner_pair) == 2:
                (ax, ay), (bx, by) = corner_pair
                dx, dy = float(bx) - float(ax), float(by) - float(ay)
                corner_norm = math.hypot(dx, dy)
                corner_span = abs(dx * depth_unit[0] + dy * depth_unit[1])
                corner_alignment = corner_span / max(corner_norm, 1e-9)
                inferred_depth = corner_span / projected_depth
                if inferred_depth > 1e-9:
                    detected_ratio = inferred_length / inferred_depth
                minimum_span = max(6.0, visible_width * 0.04)
                if (
                    corner_confidence >= 0.45
                    and corner_span >= minimum_span
                    and corner_alignment >= math.cos(math.radians(35.0))
                    and detected_ratio is not None
                    and 1.15 <= detected_ratio <= 5.0
                ):
                    depth_source = "vlm_visible_short_end_ground_corners"
            if depth_source == "vlm_visible_short_end_ground_corners":
                top_recede = corner_span
                solved_ratio = detected_ratio
            elif mask_ratio_fit["accepted"]:
                solved_ratio = float(mask_ratio_fit["ratio"])
                top_recede = (inferred_length / solved_ratio) * projected_depth
                depth_source = "sam_silhouette_fixed_yaw_ratio_fit"
            else:
                fallback_depth = inferred_length / 3.0
                top_recede = fallback_depth * projected_depth
                solved_ratio = 3.0
            wall_height = max(12.0, visible_height * 0.32)
            observations.append(
                HouseObservation(
                    instance["id"],
                    ((x1 + x2) / 2.0, min(camera.height - 1.0, ground_y)),
                    broad_span,
                    top_recede,
                    wall_height,
                    yaw_deg,
                    "front",
                    {
                        "long_dimension_source": "sam_visible_body_envelope",
                        "depth_source": depth_source,
                        "visible_short_end_ground_corners_px": corner_pair,
                        "short_end_corners_confidence": round(corner_confidence, 6),
                        "short_end_projected_span_px": round(corner_span, 3),
                        "short_end_axis_alignment": round(corner_alignment, 6),
                        "raw_detected_length_depth_ratio": (
                            round(detected_ratio, 6) if detected_ratio is not None else None
                        ),
                        "solved_length_depth_ratio_before_placement": round(solved_ratio, 6),
                        "mask_ratio_fit": mask_ratio_fit,
                    },
                )
            )
        return observations

    raise ValueError("A detection cache is required; manual house-coordinate fallback is disabled")


def solve_houses(
    observations: Sequence[HouseObservation],
    camera: Camera,
    scene: dict | None = None,
    enforce_placement_constraints: bool = True,
) -> List[House]:
    """Infer world dimensions from the gated 2D face evidence, then place."""
    houses: List[House] = []
    for obs in observations:
        relative_yaw = math.radians(obs.yaw_deg - camera.azimuth_deg)
        projected_long = camera.scale * math.hypot(
            math.cos(relative_yaw), math.sin(camera.pitch) * math.sin(relative_yaw)
        )
        projected_depth = camera.scale * math.hypot(
            math.sin(relative_yaw), math.sin(camera.pitch) * math.cos(relative_yaw)
        )
        projected_height = camera.scale * math.cos(camera.pitch)
        length = obs.broad_span_px / projected_long
        depth = obs.top_recede_px / projected_depth
        height = obs.wall_height_px / projected_height
        gx, gy = camera.ground_from_pixel(obs.ground_center_px)
        houses.append(
            House(
                obs.id,
                (gx, gy, 0.0),
                length,
                depth,
                height,
                obs.yaw_deg,
                obs.broad_face,
                obs.ground_center_px,
            )
        )
    if not enforce_placement_constraints:
        return houses
    # Enforce non-overlap and keep footprints out of detected water while
    # retaining the 2D evidence as closely as possible. Moving along world-y
    # corresponds to the ground-contact/occlusion signal.
    waters = world_water_polygons(scene, camera) if scene is not None else []
    # Resolve overlaps in the final body envelope before dimensions lock. Any
    # scale correction is strictly uniform, so the inferred footprint ratio is
    # immutable; remaining conflicts are handled by x/y placement below.
    for _ in range(28):
        collision_found = False
        for i in range(len(houses)):
            for j in range(i + 1, len(houses)):
                if polygons_overlap(houses[i].footprint(), houses[j].footprint(), tolerance=-0.01):
                    for house in (houses[i], houses[j]):
                        house.length *= 0.985
                        house.depth *= 0.985
                    collision_found = True
        if not collision_found:
            break
    for _ in range(96):
        changed = False
        for house in houses:
            if any(
                point_in_polygon(point, water["footprint"])
                for water in waters
                for point in house.footprint()
            ):
                house.center = (house.center[0], house.center[1] + 0.06, 0.0)
                changed = True
        for i in range(len(houses)):
            for j in range(i + 1, len(houses)):
                # A small negative tolerance asks SAT for a real clearance,
                # avoiding numerical "touching" that is actually a thin overlap.
                if polygons_overlap(houses[i].footprint(), houses[j].footprint(), tolerance=-0.01):
                    a, b = houses[i], houses[j]
                    # The lower source ground contact is nearer and moves toward -y.
                    near, far = (a, b) if a.source_center_px[1] > b.source_center_px[1] else (b, a)
                    near.center = (near.center[0], near.center[1] - 0.08, 0.0)
                    far.center = (far.center[0], far.center[1] + 0.08, 0.0)
                    changed = True
        if not changed:
            break
    return houses


def project_house_faces(house: House, camera: Camera) -> Dict[str, List[Point2]]:
    return {
        name: [camera.project(p) for p in poly]
        for name, poly in house.faces().items()
        if name in house.visible_faces()
    }


def jitter_raw_polygon(
    poly: Sequence[Point2], house_index: int, face_index: int, width: int, height: int
) -> List[Point2]:
    # Deterministic sub-brush-edge perturbation.  Geometry correction removes
    # these 1-4 px deviations before the cuboid solve.
    pattern = [(-2, 1), (2, 2), (1, -2), (-1, -2)]
    amount = 1.0 + ((house_index + face_index) % 3) * 0.65
    out = []
    for i, (x, y) in enumerate(poly):
        dx, dy = pattern[(i + house_index + face_index) % len(pattern)]
        out.append((clamp(x + dx * amount, 0, width - 1), clamp(y + dy * amount, 0, height - 1)))
    return out


def axes_for_polygon(poly: Sequence[Point2]) -> Iterable[Point2]:
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        ex, ey = q[0] - p[0], q[1] - p[1]
        length = math.hypot(ex, ey)
        if length > 1e-9:
            yield (-ey / length, ex / length)


def projection_interval(poly: Sequence[Point2], axis: Point2) -> Tuple[float, float]:
    values = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
    return min(values), max(values)


def polygons_overlap(a: Sequence[Point2], b: Sequence[Point2], tolerance: float = 0.0) -> bool:
    for axis in list(axes_for_polygon(a)) + list(axes_for_polygon(b)):
        amin, amax = projection_interval(a, axis)
        bmin, bmax = projection_interval(b, axis)
        if amax <= bmin + tolerance or bmax <= amin + tolerance:
            return False
    return True


def point_in_polygon(point: Point2, poly: Sequence[Point2]) -> bool:
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            cross_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def draw_nonhouses(
    image: Image.Image, scene: dict, width: int, height: int, alpha: int = 255
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for mountain in scene["mountains"]:
        draw.polygon(
            px_polygon(mountain["polygon"], width, height), fill=(*PALETTE["mountain"], alpha)
        )
    for water in scene["waters"]:
        draw.polygon(px_polygon(water["polygon"], width, height), fill=(*PALETTE["water"], alpha))
    for tree in scene["tree_masses"]:
        draw.polygon(
            px_polygon(tree["polygon"], width, height), fill=(*PALETTE["tree_mass"], alpha)
        )


def house_draw_order(houses: Sequence[House]) -> List[House]:
    # Larger world-y is farther and is painted first.
    return sorted(houses, key=lambda h: h.center[1], reverse=True)


def record_house_faces(
    object_id: str,
    broad_face: str,
    ground_contact: Point2,
    out_faces: Dict[str, List[List[List[float]]]],
) -> dict:
    all_pts = [p for polygons in out_faces.values() for polygon in polygons for p in polygon]
    return {
        "id": object_id,
        "class": "House",
        "whole_house_bbox": [
            round(min(p[0] for p in all_pts), 2),
            round(min(p[1] for p in all_pts), 2),
            round(max(p[0] for p in all_pts), 2),
            round(max(p[1] for p in all_pts), 2),
        ],
        "visible_faces": out_faces,
        "door_evidence": broad_face == "front",
        "inferred_ground_contact": [round(ground_contact[0], 2), round(ground_contact[1], 2)],
    }


def render_raw_semantic(
    scene: dict, observations: Sequence[HouseObservation], camera: Camera
) -> Tuple[Image.Image, dict]:
    """Emit visible 2D evidence before any House/mesh object is solved."""
    image = Image.new("RGB", (camera.width, camera.height), (255, 255, 255))
    draw_nonhouses(image, scene, camera.width, camera.height)
    records = []
    # Higher image ground contacts are farther and are painted first.
    ordered_observations = sorted(observations, key=lambda o: o.ground_center_px[1])
    by_id = {obs.id: i for i, obs in enumerate(observations)}
    for obs in ordered_observations:
        face_polys = obs.face_polygons(camera)
        ordered_faces = [name for name in obs.visible_faces() if name != "top"] + ["top"]
        out_faces = {}
        for face_index, face in enumerate(ordered_faces):
            poly = jitter_raw_polygon(
                face_polys[face], by_id[obs.id], face_index, camera.width, camera.height
            )
            ImageDraw.Draw(image).polygon(poly, fill=FACE_COLOR[face])
            out_faces[face] = [round_poly(poly)]
        records.append(record_house_faces(obs.id, obs.broad_face, obs.ground_center_px, out_faces))
    return image, {"houses": sorted(records, key=lambda r: r["id"])}


def render_semantic(
    scene: dict, houses: Sequence[House], camera: Camera
) -> Tuple[Image.Image, dict]:
    image = Image.new("RGB", (camera.width, camera.height), (255, 255, 255))
    draw_nonhouses(image, scene, camera.width, camera.height)
    records = []
    for house in house_draw_order(houses):
        face_polys = project_house_faces(house, camera)
        ordered = [name for name in house.visible_faces() if name != "top"] + ["top"]
        out_faces = {}
        for face in ordered:
            poly = face_polys[face]
            ImageDraw.Draw(image).polygon(poly, fill=FACE_COLOR[face])
            out_faces[face] = [round_poly(poly)]
        records.append(
            record_house_faces(house.id, house.broad_face, house.source_center_px, out_faces)
        )
    return image, {"houses": sorted(records, key=lambda r: r["id"])}


def semantic_json(
    scene_name: str, scene: dict, camera: Camera, house_records: dict, stage: str
) -> dict:
    instances = list(house_records["houses"])
    for key, class_name in (
        ("mountains", "Mountain"),
        ("waters", "Water"),
        ("tree_masses", "TreeMass"),
    ):
        for item in scene[key]:
            instances.append(
                {
                    "id": item["id"],
                    "class": class_name,
                    "polygons": [
                        round_poly(px_polygon(item["polygon"], camera.width, camera.height))
                    ],
                }
            )
    valid_ids = {item["id"] for item in instances}
    return {
        "scene_id": scene_name,
        "stage": stage,
        "image_size": [camera.width, camera.height],
        "coordinate_system": "pixel_xy_origin_top_left",
        "palette": PALETTE_HEX,
        "instances": instances,
        "occlusion_relations": [
            {"front_id": front, "behind_id": behind, "confidence": conf}
            for front, behind, conf in scene["occlusion"]
            if front in valid_ids and behind in valid_ids
        ],
    }


def draw_instances(original: Image.Image, scene: dict, semantic: dict) -> Image.Image:
    overlay = original.convert("RGBA")
    tint = Image.new("RGBA", original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tint, "RGBA")
    for item in semantic["instances"]:
        if item["class"] == "House":
            polys = [poly for polys in item["visible_faces"].values() for poly in polys]
            color = (230, 75, 53, 78)
        else:
            polys = item["polygons"]
            color_key = {"Mountain": "mountain", "Water": "water", "TreeMass": "tree_mass"}[
                item["class"]
            ]
            color = (*PALETTE[color_key], 55)
        for poly in polys:
            draw.polygon(poly, fill=color, outline=(*color[:3], 210), width=2)
        points = [p for poly in polys for p in poly]
        if points:
            x, y = min(p[0] for p in points), min(p[1] for p in points)
            label = f"{item['id']}  {item['class']}"
            box = draw.textbbox((x + 3, y + 3), label, font=ImageFont.load_default())
            draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(20, 20, 20, 205))
            draw.text(
                (x + 3, y + 3), label, fill=(255, 255, 255, 255), font=ImageFont.load_default()
            )
    return Image.alpha_composite(overlay, tint).convert("RGB")


def load_automatic_house_instances(
    scene_name: str, image_size: Tuple[int, int], detection_dir: Path
) -> Tuple[dict, Image.Image] | None:
    """Load the audited detector/classifier/SAM parsing gate when available."""
    json_path = detection_dir / f"{scene_name}_building_instances.json"
    overlay_path = detection_dir / f"{scene_name}_building_instances.png"
    mask_path = detection_dir / f"{scene_name}_building_instance_mask.png"
    if not (json_path.is_file() and overlay_path.is_file() and mask_path.is_file()):
        return None
    document = json.loads(json_path.read_text(encoding="utf-8"))
    if tuple(document.get("image_size", ())) != image_size:
        raise ValueError(f"{json_path}: image size does not match reconstruction input")
    if document.get("stage") != "automatic_whole_building_instance_parsing":
        raise ValueError(f"{json_path}: unexpected parsing stage")
    identifiers = [row.get("id") for row in document.get("instances", [])]
    if not identifiers or len(identifiers) != len(set(identifiers)) or not all(identifiers):
        raise ValueError(f"{json_path}: missing, empty or duplicate instance IDs")
    if document.get("count") != len(identifiers):
        raise ValueError(f"{json_path}: instance count differs from the actual list")
    overlay = Image.open(overlay_path).convert("RGB")
    if overlay.size != image_size:
        raise ValueError(f"{overlay_path}: overlay size does not match reconstruction input")
    return document, overlay


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_automatic_building_orientations(
    scene_name: str,
    image_path: Path,
    image_size: Tuple[int, int],
    detection_dir: Path,
    orientation_dir: Path,
) -> Tuple[dict, Image.Image] | None:
    """Load the second-pass VLM cache and reject stale detector/image bindings."""
    json_path = orientation_dir / f"{scene_name}_building_orientations.json"
    overlay_path = orientation_dir / f"{scene_name}_building_orientations.png"
    detection_path = detection_dir / f"{scene_name}_building_instances.json"
    if not (json_path.is_file() and overlay_path.is_file()):
        return None
    document = json.loads(json_path.read_text(encoding="utf-8"))
    if document.get("stage") != "automatic_building_orientation_inference":
        raise ValueError(f"{json_path}: unexpected orientation stage")
    if tuple(document.get("image_size", ())) != image_size:
        raise ValueError(f"{json_path}: image size does not match reconstruction input")
    if document.get("image_sha256") != file_sha256(image_path):
        raise ValueError(f"{json_path}: painting hash is stale")
    if document.get("detection_sha256") != file_sha256(detection_path):
        raise ValueError(f"{json_path}: detector-document hash is stale")
    overlay = Image.open(overlay_path).convert("RGB")
    if overlay.size != image_size:
        raise ValueError(f"{overlay_path}: overlay size does not match reconstruction input")
    return document, overlay


def render_normal(scene: dict, houses: Sequence[House], camera: Camera) -> Image.Image:
    image = Image.new("RGB", (camera.width, camera.height), (225, 219, 193))
    draw = ImageDraw.Draw(image, "RGBA")
    for mountain in scene["mountains"]:
        draw.polygon(
            px_polygon(mountain["polygon"], camera.width, camera.height), fill=(116, 139, 91, 255)
        )
    for water in scene["waters"]:
        draw.polygon(
            px_polygon(water["polygon"], camera.width, camera.height), fill=(104, 151, 190, 255)
        )
    for tree in scene["tree_masses"]:
        draw.polygon(
            px_polygon(tree["polygon"], camera.width, camera.height), fill=(70, 104, 62, 145)
        )
    wall_base = (190, 153, 100)
    for house in house_draw_order(houses):
        faces = project_house_faces(house, camera)
        for face in [f for f in house.visible_faces() if f != "top"]:
            factor = 1.05 if face in ("front", "back") else 0.78
            draw.polygon(
                faces[face],
                fill=(*shade(wall_base, factor), 255),
                outline=(80, 60, 43, 255),
                width=2,
            )
        draw.polygon(faces["top"], fill=(88, 77, 56, 255), outline=(53, 45, 33, 255), width=2)
    return image


def world_water_polygons(scene: dict, camera: Camera) -> List[dict]:
    result = []
    for water in scene["waters"]:
        poly = [
            camera.ground_from_pixel(p)
            for p in px_polygon(water["polygon"], camera.width, camera.height)
        ]
        result.append({"id": water["id"], "footprint": poly})
    return result


def render_top(scene: dict, houses: Sequence[House], camera: Camera) -> Image.Image:
    width, height = camera.width, camera.height
    waters = world_water_polygons(scene, camera)
    mountain_footprints = []
    for mountain in scene["mountains"]:
        screen_poly = px_polygon(mountain["polygon"], width, height)
        # A coarse ground footprint spanning the visible base horizontally.
        xs = [p[0] for p in screen_poly]
        foot_y = mountain.get("foot_y", 0.7) * height
        near = camera.ground_from_pixel(((min(xs) + max(xs)) / 2, foot_y))
        span = (max(xs) - min(xs)) / camera.scale
        mountain_footprints.append(
            {
                "id": mountain["id"],
                "footprint": [
                    (near[0] - span / 2, near[1]),
                    (near[0] + span / 2, near[1]),
                    (near[0] + span / 2, near[1] + 2.2),
                    (near[0] - span / 2, near[1] + 2.2),
                ],
            }
        )
    tree_points = []
    for tree in scene["tree_masses"]:
        fp = (tree["foot"][0] * width, tree["foot"][1] * height)
        tree_points.append((tree["id"], camera.ground_from_pixel(fp)))

    all_points = [p for h in houses for p in h.footprint()]
    all_points += [p for w in waters for p in w["footprint"]]
    all_points += [p for m in mountain_footprints for p in m["footprint"]]
    all_points += [p for _, p in tree_points]
    min_x, max_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    pad_x, pad_y = max(0.8, (max_x - min_x) * 0.06), max(0.8, (max_y - min_y) * 0.06)
    min_x, max_x, min_y, max_y = min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y
    scale = min((width - 70) / (max_x - min_x), (height - 70) / (max_y - min_y))
    ox = (width - (max_x - min_x) * scale) / 2 - min_x * scale
    oy = (height - (max_y - min_y) * scale) / 2 + max_y * scale

    def top_px(p: Point2) -> Point2:
        return ox + p[0] * scale, oy - p[1] * scale

    image = Image.new("RGB", (width, height), (224, 222, 196))
    draw = ImageDraw.Draw(image, "RGBA")
    for m in mountain_footprints:
        draw.polygon(
            [top_px(p) for p in m["footprint"]],
            fill=(122, 150, 108, 210),
            outline=(74, 102, 72, 255),
            width=2,
        )
    for w in waters:
        draw.polygon(
            [top_px(p) for p in w["footprint"]],
            fill=(118, 165, 216, 230),
            outline=(63, 105, 154, 255),
            width=2,
        )
    for tree_id, p in tree_points:
        x, y = top_px(p)
        r = 0.24 * scale
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(75, 112, 65, 180),
            outline=(48, 78, 46, 255),
            width=2,
        )
        draw.text(
            (x + r + 2, y - 5), tree_id, fill=(35, 49, 33, 255), font=ImageFont.load_default()
        )
    for house in houses:
        poly = [top_px(p) for p in house.footprint()]
        draw.polygon(poly, fill=(196, 158, 103, 255), outline=(62, 50, 38, 255), width=3)
        # Mark the local front edge so independent yaw is auditable.
        face = house.faces()["front"][:2]
        front_edge = [top_px(p[:2]) for p in face]
        draw.line(front_edge, fill=(*PALETTE["house_front"], 255), width=5)
        cx, cy = top_px(house.center[:2])
        direction_world = house.point(house.length * 0.38, 0.0, 0.0)
        ex, ey = top_px(direction_world[:2])
        draw.line((cx, cy, ex, ey), fill=(35, 31, 27, 255), width=4)
        angle = math.atan2(ey - cy, ex - cx)
        for wing in (-2.5, 2.5):
            draw.line(
                (ex, ey, ex + math.cos(angle + wing) * 11, ey + math.sin(angle + wing) * 11),
                fill=(35, 31, 27, 255),
                width=4,
            )
        draw.text(
            (cx - 25, cy - 6), house.id, fill=(30, 25, 22, 255), font=ImageFont.load_default()
        )
    # Orientation legend required by the output contract.
    draw.rectangle((12, 12, 234, 70), fill=(250, 250, 244, 225), outline=(60, 60, 55, 255), width=1)
    draw.text(
        (22, 20),
        "PURE ORTHOGRAPHIC TOP (90 deg)",
        fill=(20, 20, 20, 255),
        font=ImageFont.load_default(),
    )
    draw.text(
        (22, 38), "painting far / upper", fill=(20, 20, 20, 255), font=ImageFont.load_default()
    )
    draw.line((180, 59, 180, 38), fill=(20, 20, 20, 255), width=2)
    draw.polygon([(176, 42), (180, 36), (184, 42)], fill=(20, 20, 20, 255))
    draw.text(
        (22, 54),
        "dark arrow = directed visible end",
        fill=(20, 20, 20, 255),
        font=ImageFont.load_default(),
    )
    return image


def render_perspective_debug(scene: dict, houses: Sequence[House], camera: Camera) -> Image.Image:
    # A deliberately separate, clearly labelled debug view.  The contracted top
    # view above never calls this perspective projection.
    width, height = camera.width, camera.height
    image = Image.new("RGB", (width, height), (216, 220, 210))
    draw = ImageDraw.Draw(image)
    eye = (0.0, -19.0, 13.0)
    target = (0.0, 3.0, 0.0)
    forward = tuple(target[i] - eye[i] for i in range(3))
    fl = math.sqrt(sum(v * v for v in forward))
    forward = tuple(v / fl for v in forward)
    right = (forward[1], -forward[0], 0.0)
    rl = math.hypot(right[0], right[1])
    right = (right[0] / rl, right[1] / rl, 0.0)
    up = (
        right[1] * forward[2],
        -right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )

    def project(p: Point3) -> Point2:
        v = tuple(p[i] - eye[i] for i in range(3))
        depth = sum(v[i] * forward[i] for i in range(3))
        sx = sum(v[i] * right[i] for i in range(3)) / max(depth, 0.1)
        sy = sum(v[i] * up[i] for i in range(3)) / max(depth, 0.1)
        return width / 2 + sx * 920, height * 0.62 - sy * 920

    for house in sorted(houses, key=lambda h: h.center[1], reverse=True):
        faces = house.faces()
        for face in (house.broad_face, "left" if house.yaw_deg > 0 else "right", "top"):
            color = (181, 143, 94) if face != "top" else (83, 72, 54)
            draw.polygon(
                [project(p) for p in faces[face]], fill=color, outline=(50, 44, 36), width=2
            )
    draw.rectangle((12, 12, 235, 40), fill=(255, 248, 230), outline=(60, 60, 55))
    draw.text((20, 21), "PERSPECTIVE DEBUG ONLY", fill=(140, 30, 25), font=ImageFont.load_default())
    return image


def mask_for_colors(image: Image.Image, colors: Iterable[Tuple[int, int, int]]) -> bytearray:
    wanted = set(colors)
    pixels = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
    return bytearray(1 if p in wanted else 0 for p in pixels)


def mask_iou(a: bytearray, b: bytearray) -> float:
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return 1.0 if union == 0 else inter / union


def correction_ratio(raw: Image.Image, geometry: Image.Image) -> float:
    colors = FACE_COLOR.values()
    a, b = mask_for_colors(raw, colors), mask_for_colors(geometry, colors)
    changed = sum(1 for x, y in zip(a, b) if x != y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return 0.0 if union == 0 else changed / union


def validate_constraints(scene: dict, houses: Sequence[House], camera: Camera) -> dict:
    violations = []
    for house in houses:
        if not house.length > house.depth:
            violations.append(f"{house.id}: length <= depth")
        if abs(house.center[2]) > 1e-9:
            violations.append(f"{house.id}: z != 0")
        visible = house.visible_faces()
        if "front" in visible and "back" in visible:
            violations.append(f"{house.id}: front and back both visible")
        if "left" in visible and "right" in visible:
            violations.append(f"{house.id}: left and right both visible")
    for i in range(len(houses)):
        for j in range(i + 1, len(houses)):
            if polygons_overlap(houses[i].footprint(), houses[j].footprint(), tolerance=0.01):
                violations.append(f"{houses[i].id}<->{houses[j].id}: solid overlap")
    waters = world_water_polygons(scene, camera)
    for house in houses:
        for water in waters:
            if any(point_in_polygon(p, water["footprint"]) for p in house.footprint()):
                violations.append(f"{house.id}<->{water['id']}: footprint in water")
    return {"count": len(violations), "items": violations}


class GLBBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.material_by_color = {}

    def align4(self) -> None:
        while len(self.binary) % 4:
            self.binary.append(0)

    def material(self, color: Tuple[int, int, int], alpha: float = 1.0) -> int:
        key = (color, alpha)
        if key in self.material_by_color:
            return self.material_by_color[key]
        idx = len(self.materials)
        self.materials.append(
            {
                "name": f"Material_{idx:02d}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [color[0] / 255, color[1] / 255, color[2] / 255, alpha],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.88,
                },
                "doubleSided": True,
                **({"alphaMode": "BLEND"} if alpha < 1 else {}),
            }
        )
        self.material_by_color[key] = idx
        return idx

    def add_primitive(
        self, positions: Sequence[Point3], indices: Sequence[int], material: int
    ) -> dict:
        self.align4()
        pos_offset = len(self.binary)
        for p in positions:
            self.binary += struct.pack("<fff", *p)
        pos_length = len(self.binary) - pos_offset
        pos_view = len(self.buffer_views)
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_length, "target": 34962}
        )
        mins = [min(p[i] for p in positions) for i in range(3)]
        maxs = [max(p[i] for p in positions) for i in range(3)]
        pos_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            }
        )
        self.align4()
        idx_offset = len(self.binary)
        for value in indices:
            self.binary += struct.pack("<H", value)
        idx_length = len(self.binary) - idx_offset
        idx_view = len(self.buffer_views)
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_length, "target": 34963}
        )
        idx_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": idx_view,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
                "min": [min(indices)],
                "max": [max(indices)],
            }
        )
        return {
            "attributes": {"POSITION": pos_accessor},
            "indices": idx_accessor,
            "material": material,
            "mode": 4,
        }

    def add_mesh(self, name: str, primitives: Sequence[dict]) -> None:
        mesh_idx = len(self.meshes)
        self.meshes.append({"name": name, "primitives": list(primitives)})
        self.nodes.append({"name": name, "mesh": mesh_idx})

    def write(self, path: Path) -> None:
        self.align4()
        doc = {
            "asset": {"version": "2.0", "generator": "ancient3d dependency-light pipeline"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "buffers": [{"byteLength": len(self.binary)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
        }
        json_chunk = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        while len(json_chunk) % 4:
            json_chunk += b" "
        bin_chunk = bytes(self.binary)
        total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        payload = struct.pack("<4sII", b"glTF", 2, total)
        payload += struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
        payload += struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
        path.write_bytes(payload)


def cuboid_primitives(builder: GLBBuilder, house: House) -> List[dict]:
    faces = house.faces()
    primitives = []
    wall_material = builder.material((190, 153, 100))
    top_material = builder.material((88, 77, 56))
    for face in ("front", "back", "left", "right", "top"):
        poly = faces[face]
        primitives.append(
            builder.add_primitive(
                poly, [0, 1, 2, 0, 2, 3], top_material if face == "top" else wall_material
            )
        )
    return primitives


def export_glb(path: Path, scene: dict, houses: Sequence[House], camera: Camera) -> None:
    builder = GLBBuilder()
    for house in houses:
        builder.add_mesh(house.id, cuboid_primitives(builder, house))
    for water in world_water_polygons(scene, camera):
        poly = [(x, y, 0.005) for x, y in water["footprint"]]
        if len(poly) >= 3:
            indices = []
            for i in range(1, len(poly) - 1):
                indices += [0, i, i + 1]
            builder.add_mesh(
                water["id"],
                [builder.add_primitive(poly, indices, builder.material((118, 165, 216), 0.88))],
            )
    for mountain in scene["mountains"]:
        screen_poly = px_polygon(mountain["polygon"], camera.width, camera.height)
        min_x, max_x = min(p[0] for p in screen_poly), max(p[0] for p in screen_poly)
        foot_y = mountain.get("foot_y", 0.7) * camera.height
        center = camera.ground_from_pixel(((min_x + max_x) / 2, foot_y))
        width = max(1.2, (max_x - min_x) / camera.scale)
        depth = 2.2
        base = [
            (center[0] - width / 2, center[1], 0),
            (center[0] + width / 2, center[1], 0),
            (center[0] + width / 2, center[1] + depth, 0),
            (center[0] - width / 2, center[1] + depth, 0),
        ]
        apex = (center[0], center[1] + depth * 0.55, max(1.3, width * 0.28))
        positions = base + [apex]
        indices = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 3, 2, 0, 2, 1]
        builder.add_mesh(
            mountain["id"],
            [builder.add_primitive(positions, indices, builder.material((116, 139, 91)))],
        )
    for tree in scene["tree_masses"]:
        fp = (tree["foot"][0] * camera.width, tree["foot"][1] * camera.height)
        x, y = camera.ground_from_pixel(fp)
        r, h = 0.055, 1.55
        positions = []
        segments = 8
        for z in (0.0, h):
            for i in range(segments):
                angle = i * 2 * math.pi / segments
                positions.append((x + r * math.cos(angle), y + r * math.sin(angle), z))
        indices = []
        for i in range(segments):
            j = (i + 1) % segments
            indices += [i, j, segments + j, i, segments + j, segments + i]
        trunk = builder.add_primitive(positions, indices, builder.material((78, 59, 39)))
        crown_positions = [
            (x - 0.38, y, h - 0.20),
            (x + 0.38, y, h - 0.20),
            (x + 0.38, y, h + 0.55),
            (x - 0.38, y, h + 0.55),
            (x, y - 0.38, h - 0.20),
            (x, y + 0.38, h - 0.20),
            (x, y + 0.38, h + 0.55),
            (x, y - 0.38, h + 0.55),
        ]
        crown = builder.add_primitive(
            crown_positions,
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
            builder.material((70, 104, 62), 0.58),
        )
        builder.add_mesh(tree["id"], [trunk, crown])
    builder.write(path)


def scene_json(scene_name: str, scene: dict, houses: Sequence[House], camera: Camera) -> dict:
    valid_ids = {house.id for house in houses}
    valid_ids.update(
        item["id"] for key in ("mountains", "waters", "tree_masses") for item in scene[key]
    )
    return {
        "scene_id": scene_name,
        "units": "arbitrary_relative_units",
        "coordinate_system": {"x": "painting_right", "y": "painting_far", "z": "up"},
        "camera": {
            "projection": "orthographic",
            "azimuth_deg": camera.azimuth_deg,
            "pitch_deg": camera.pitch_deg,
            "roll_deg": 0,
            "ortho_scale": camera.scale,
            "image_offset": [round(camera.offset_x, 3), round(camera.offset_y, 3)],
        },
        "houses": [
            {
                "id": h.id,
                "center": [round(v, 5) for v in h.center],
                "length": h.length,
                "depth": h.depth,
                "height": h.height,
                "yaw_deg": h.yaw_deg,
                "pitch_deg": 0,
                "roll_deg": 0,
                "dimensions_locked": True,
                "visible_faces": h.visible_faces(),
                "evidence": {
                    "whole_house_detected_first": True,
                    "ground_contact_px": [round(v, 2) for v in h.source_center_px],
                    "broad_face_classification": (
                        "automatic_visible_region_front_axis"
                        if h.broad_face == "front"
                        else "visible_broad_face_defaults_back"
                    ),
                    "long_dimension_source": "visible_broad_face",
                    "depth_source": "visible_top_and_side",
                    "height_source": "visible_vertical_face",
                },
            }
            for h in houses
        ],
        "mountains": [
            {
                "id": m["id"],
                "geometry": "coarse_pyramidal_mass",
                "source_polygon_normalized": m["polygon"],
            }
            for m in scene["mountains"]
        ],
        "waters": [
            {
                "id": w["id"],
                "geometry": "horizontal_plane",
                "z": 0,
                "source_polygon_normalized": w["polygon"],
            }
            for w in scene["waters"]
        ],
        "trees": [
            {
                "id": t["id"],
                "class": "TreeMass",
                "geometry": "trunk_plus_transparent_crossed_crown",
                "crown_occludes_house_semantics": False,
            }
            for t in scene["tree_masses"]
        ],
        "occlusion_relations": [
            {"front_id": front, "behind_id": behind, "confidence": confidence}
            for front, behind, confidence in scene["occlusion"]
            if front in valid_ids and behind in valid_ids
        ],
        "top_view_camera": {
            "projection": "orthographic",
            "elevation_deg": 90,
            "maps": {"world_x": "image_right", "world_y": "image_up", "world_z": "view_depth"},
        },
    }


def make_comparison(
    scene_name: str, panels: Sequence[Tuple[str, Image.Image]], width: int
) -> Image.Image:
    cols = 4
    thumb_w = width // cols
    thumb_h = int(thumb_w * 0.78)
    header = 44
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (thumb_w * cols, rows * (thumb_h + header)), (238, 235, 224))
    draw = ImageDraw.Draw(sheet)
    for i, (title, image) in enumerate(panels):
        col, row = i % cols, i // cols
        x, y = col * thumb_w, row * (thumb_h + header)
        fitted = image.copy()
        fitted.thumbnail((thumb_w - 10, thumb_h - 10), Image.Resampling.LANCZOS)
        px = x + (thumb_w - fitted.width) // 2
        py = y + header + (thumb_h - fitted.height) // 2
        sheet.paste(fitted, (px, py))
        draw.rectangle((x, y, x + thumb_w - 1, y + header - 1), fill=(42, 48, 43))
        draw.text(
            (x + 9, y + 8),
            f"{scene_name} | {title}",
            fill=(255, 255, 255),
            font=ImageFont.load_default(),
        )
    return sheet


def process_scene(
    input_path: Path, out_root: Path, scene: dict, detection_dir: Path, orientation_dir: Path
) -> Tuple[Path, dict]:
    scene_name = input_path.stem
    original = Image.open(input_path).convert("RGB")
    width, height = original.size
    cam_cfg = scene["camera"]
    camera = Camera(
        width,
        height,
        cam_cfg["pitch_deg"],
        cam_cfg["azimuth_deg"],
        cam_cfg["scale"],
        cam_cfg["offset_y"],
    )
    out_dir = out_root / scene_name
    # Validate cache bindings before creating any scene output.
    automatic_detection = load_automatic_house_instances(scene_name, original.size, detection_dir)
    detection_doc = automatic_detection[0] if automatic_detection is not None else None
    if automatic_detection is None:
        raise ValueError(f"Missing complete detection cache for {scene_name}")
    automatic_orientation = load_automatic_building_orientations(
        scene_name, input_path, original.size, detection_dir, orientation_dir
    )
    orientation_doc = automatic_orientation[0] if automatic_orientation is not None else None
    if detection_doc is not None and orientation_doc is None:
        raise ValueError(
            f"Missing second-pass building orientation cache for {scene_name}; "
            "run infer_building_orientations_vlm.py after instance detection"
        )
    observations = parse_house_observations(scene, camera, detection_doc, orientation_doc)
    out_dir.mkdir(parents=True, exist_ok=False)
    original.save(out_dir / "original.png")
    raw, raw_records = render_raw_semantic(scene, observations, camera)
    raw_doc = semantic_json(
        scene_name, scene, camera, raw_records, "cuboid_derived_preliminary_semantics"
    )
    raw_doc["independent_face_segmentation"] = False
    raw_doc["synthetic_boundary_jitter"] = True
    raw_doc["scene_priors"] = "Explicit camera/terrain assumptions supplied in scene configuration"
    if automatic_detection is not None:
        instances = (
            automatic_orientation[1]
            if automatic_orientation is not None
            else automatic_detection[1]
        )
        model_provenance = dict(detection_doc["models"])
        model_provenance["independent_orientation_model"] = orientation_doc["model"]
        raw_doc["automatic_whole_house_parsing"] = {
            "source": f"house_detection/{scene_name}_building_instances.json",
            "method": "Qwen3-VL-8B multiscale whole-building grounding + YOLO child evidence -> SAM 2.1 -> RGB classifier -> compound parent decomposition -> containment and structure filters; then an independent larger VLM pass infers directed per-building orientation",
            "models": model_provenance,
            "settings": detection_doc["settings"],
            "detected_count": detection_doc["count"],
            "cuboid_observation_count": len(observations),
            "count_matches": detection_doc["count"] == len(observations),
            "geometry_observations_derived_from_detection": True,
            "orientation_source": str(
                Path("building_orientation") / f"{scene_name}_building_orientations.json"
            ),
            "orientation_method": orientation_doc["method"],
            "orientation_definition": orientation_doc["orientation_definition"],
            "orientation_count_matches": len(orientation_doc["orientations"]) == len(observations),
        }
    else:
        instances = draw_instances(original, scene, raw_doc)
    raw.save(out_dir / "raw_semantic.png")
    (out_dir / "raw_semantic.json").write_text(json.dumps(raw_doc, indent=2), encoding="utf-8")
    instances.save(out_dir / "instances.png")

    # Only after the parsing gate do 2D measurements become world dimensions.
    cuboid_reference_houses = solve_houses(
        observations,
        camera,
        scene,
        enforce_placement_constraints=False,
    )
    houses = solve_houses(observations, camera, scene)

    # Both target and render below are generated from the same fitted cuboids.
    # Their agreement is a renderer consistency check, not image accuracy.
    geometry, geometry_records = render_semantic(scene, houses, camera)
    geometry_doc = semantic_json(
        scene_name, scene, camera, geometry_records, "geometry_consistent_cuboid_reference"
    )
    geometry_doc["correction_policy"] = {
        "internal_boundaries_first": True,
        "outer_silhouette_preserved": False,
        "method": "cuboid-derived panel under supplied camera; placement constraints may move/scale bodies",
    }
    geometry.save(out_dir / "geometry_consistent_semantic.png")
    (out_dir / "geometry_consistent_semantic.json").write_text(
        json.dumps(geometry_doc, indent=2), encoding="utf-8"
    )

    # Stage 3: renderer uses the locked solve, not painting texture.
    semantic_render, _ = render_semantic(scene, houses, camera)
    normal_render = render_normal(scene, houses, camera)
    top_view = render_top(scene, houses, camera)
    perspective = render_perspective_debug(scene, houses, camera)
    semantic_render.save(out_dir / "original_view_semantic_render.png")
    normal_render.save(out_dir / "original_view_normal_render.png")
    top_view.save(out_dir / "top_view.png")
    perspective.save(out_dir / "perspective_debug.png")

    scene_doc = scene_json(scene_name, scene, houses, camera)
    observation_by_id = {observation.id: observation for observation in observations}
    reference_house_by_id = {house.id: house for house in cuboid_reference_houses}
    for house_record in scene_doc["houses"]:
        house_record["evidence"].update(observation_by_id[house_record["id"]].dimension_evidence)
        house_record["evidence"]["final_length_depth_ratio"] = round(
            house_record["length"] / house_record["depth"], 6
        )
        reference_house = reference_house_by_id[house_record["id"]]
        length_scale = house_record["length"] / reference_house.length
        depth_scale = house_record["depth"] / reference_house.depth
        house_record["evidence"]["prelock_uniform_body_scale"] = round(length_scale, 6)
        house_record["evidence"]["uniform_scale_preserved_ratio"] = math.isclose(
            length_scale,
            depth_scale,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    if orientation_doc is not None:
        orientation_by_id = {row["id"]: row for row in orientation_doc["orientations"]}
        for house_record in scene_doc["houses"]:
            row = orientation_by_id[house_record["id"]]
            house_record["evidence"].update(
                {
                    "yaw_source": "independent_second_pass_vlm_visible_short_end_direction",
                    "top_view_clock_direction": row["clock"],
                    "orientation_confidence": row["confidence"],
                    "orientation_model": orientation_doc["model"],
                }
            )
        scene_doc["automatic_building_orientation"] = {
            "source": str(
                Path("building_orientation") / f"{scene_name}_building_orientations.json"
            ),
            "model": orientation_doc["model"],
            "method": orientation_doc["method"],
        }
    (out_dir / "scene.json").write_text(json.dumps(scene_doc, indent=2), encoding="utf-8")
    export_glb(out_dir / "scene.glb", scene, houses, camera)

    constraints = validate_constraints(scene, houses, camera)
    geometry_iou = mask_iou(
        mask_for_colors(geometry, FACE_COLOR.values()),
        mask_for_colors(semantic_render, FACE_COLOR.values()),
    )
    cuboid_reference, _ = render_semantic(scene, cuboid_reference_houses, camera)
    raw_regularization_change = correction_ratio(raw, cuboid_reference)
    raw_final_change = correction_ratio(raw, geometry)
    metrics = {
        "scene_id": scene_name,
        "parsing_gate": {
            "passed": (
                detection_doc is not None
                and orientation_doc is not None
                and detection_doc["count"] == len(houses)
                and len(orientation_doc["orientations"]) == len(houses)
            ),
            "source": "automatic_detector_classifier_sam_plus_independent_vlm_orientation",
            "geometry_observations_source": "automatic_mask_bbox_plus_second_pass_vlm_direction",
            "whole_house_instances": len(houses),
            "automatic_detected_instances": detection_doc["count"] if detection_doc else 0,
            "automatic_oriented_instances": len(orientation_doc["orientations"])
            if orientation_doc
            else 0,
            "compound_split": len(houses) > 1,
            "face_visibility_rules_passed": True,
        },
        "geometry_gate": {
            "passed": raw_regularization_change < 0.16,
            "raw_to_cuboid_regularization_silhouette_change_ratio": round(
                raw_regularization_change, 6
            ),
            "raw_to_final_constraint_solved_silhouette_change_ratio": round(raw_final_change, 6),
            "length_greater_than_depth_all_houses": all(h.length > h.depth for h in houses),
        },
        "fit": {
            "interpretation": "self_consistency_only_same_geometry_renderer",
            "source_image_accuracy_measured": False,
            "house_semantic_silhouette_iou": round(geometry_iou, 6),
            "per_house_and_face_accuracy": None,
        },
        "hard_constraints": constraints,
        "camera": {"projection": "orthographic", "roll_deg": 0},
        "top_view": {
            "projection": "orthographic",
            "camera_elevation_degrees": 90,
            "painting_left_maps_left": True,
            "painting_far_maps_top": True,
        },
    }
    metrics["accepted"] = (
        metrics["geometry_gate"]["passed"] and constraints["count"] == 0 and geometry_iou > 0.99
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = make_comparison(
        scene_name,
        [
            ("1 original", original),
            ("2 instances", instances),
            ("3 derived semantic", raw),
            ("4 cuboid reference", geometry),
            ("5 same-geometry render", semantic_render),
            ("6 normal render", normal_render),
            ("7 true top", top_view),
            ("8 perspective debug", perspective),
        ],
        width=1600,
    )
    comparison.save(out_dir / "comparison_sheet.png")
    return out_dir, metrics


def make_batch_sheet(scene_dirs: Sequence[Path], out_root: Path) -> None:
    sheets = [
        (path.name, Image.open(path / "comparison_sheet.png").convert("RGB")) for path in scene_dirs
    ]
    width = max(image.width for _, image in sheets)
    header = 42
    height = sum(image.height + header for _, image in sheets)
    batch = Image.new("RGB", (width, height), (228, 226, 216))
    draw = ImageDraw.Draw(batch)
    y = 0
    for name, image in sheets:
        draw.rectangle((0, y, width, y + header), fill=(28, 34, 31))
        draw.text((14, y + 13), name, fill=(255, 255, 255), font=ImageFont.load_default())
        batch.paste(image, (0, y + header))
        y += header + image.height
    batch.save(out_root / "all_scenes_comparison.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="optional single input; otherwise all configured scenes"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-config", type=Path, required=True, help="JSON scene priors, no house coordinates"
    )
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--orientations-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new or empty output directory"
    )
    args = parser.parse_args()
    out_root = args.output_dir.resolve()
    if out_root.exists() and (not out_root.is_dir() or any(out_root.iterdir())):
        parser.error("Output must be a new or empty directory; nothing will be deleted")
    scenes = json.loads(args.scene_config.read_text(encoding="utf-8"))["scenes"]
    if not scenes or any("houses" in scene for scene in scenes.values()):
        parser.error("Scene configuration must contain priors, not fallback house coordinates")
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = (
        [args.input.resolve()]
        if args.input
        else [args.input_dir / (name + ".png") for name in sorted(scenes)]
    )
    scene_dirs = []
    all_ok = True
    for input_path in inputs:
        if input_path.stem not in scenes:
            raise SystemExit(f"No explicit scene configuration for {input_path.name}")
        scene_dir, metrics = process_scene(
            input_path,
            out_root,
            scenes[input_path.stem],
            args.detections_dir,
            args.orientations_dir,
        )
        scene_dirs.append(scene_dir)
        all_ok &= metrics["accepted"]
        print(f"{input_path.name}: {'PASS' if metrics['accepted'] else 'FAIL'} -> {scene_dir}")
        if metrics["hard_constraints"]["items"]:
            for item in metrics["hard_constraints"]["items"]:
                print(f"  constraint: {item}")
    if len(scene_dirs) > 1:
        make_batch_sheet(scene_dirs, out_root)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
