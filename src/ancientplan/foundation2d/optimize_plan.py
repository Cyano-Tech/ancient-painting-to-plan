"""Bounded 2D layout fitting, not an image detector or metric reconstruction.

Only plan centers can move. Identity, source geometry, sizes, facing, terrain
and source provenance are never changed. If constraints cannot be satisfied,
the solver records that fact instead of deleting/shrinking buildings.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .complete_plan import read, save, stamp
from .render_complete_plan import contains, rect_corners, overlap, geometry_checks


def feasible_centers(b, zone, boundary, max_shift, step=2):
    x, y = b["plan_center"]
    options = []
    for dx in range(-max_shift, max_shift + 1, step):
        for dy in range(-max_shift, max_shift + 1, step):
            distance = dx * dx + dy * dy
            if distance > max_shift * max_shift:
                continue
            candidate = {**b, "plan_center": [x + dx, y + dy]}
            corners = rect_corners(candidate)
            if all(contains(p, zone) and contains(p, boundary) for p in corners):
                options.append((distance, [x + dx, y + dy], corners))
    return sorted(options, key=lambda o: (o[0], o[1]))


def optimize(plan, max_shift=40):
    original = deepcopy(plan)
    buildings = original["layout"]["buildings"]
    t = original["terrain"]
    zones = {z["id"]: z["plan_polygon"] for z in t["zones"]}
    options = {
        b["id"]: feasible_centers(b, zones[b["zone_id"]], t["plan_boundary"], max_shift)
        for b in buildings
    }
    lookup = {b["id"]: b for b in buildings}
    best = None
    # Different deterministic allocation orders avoid a single greedy winner.
    orders = [
        sorted(buildings, key=lambda b: -b["plan_size"][0] * b["plan_size"][1]),
        sorted(buildings, key=lambda b: len(options[b["id"]])),
        buildings,
        list(reversed(buildings)),
    ]
    for order in orders:
        placed = {}
        failures = []
        cost = 0
        for b in order:
            chosen = None
            for distance, center, corners in options[b["id"]]:
                if any(overlap(corners, p[1]) for p in placed.values()):
                    continue
                # Preserve clear pairwise left/right and near/far ordering of
                # the supplied plan, with a tolerance for nearly aligned rows.
                inversion = False
                for oid, (other_center, _) in placed.items():
                    other = lookup[oid]
                    if other["zone_id"] != b["zone_id"]:
                        continue
                    for axis in (0, 1):
                        delta = b["plan_center"][axis] - other["plan_center"][axis]
                        if abs(delta) > 12 and (center[axis] - other_center[axis]) * delta < 0:
                            inversion = True
                if inversion:
                    continue
                chosen = (center, corners)
                cost += distance
                break
            if chosen is None:
                failures.append(b["id"])
                chosen = (b["plan_center"], rect_corners(b))
                cost += 1e9
            placed[b["id"]] = chosen
        score = (len(failures), cost)
        if best is None or score < best[0]:
            best = (score, placed, failures)
    output = deepcopy(original)
    changes = []
    for b in output["layout"]["buildings"]:
        center = best[1][b["id"]][0]
        if center != b["plan_center"]:
            changes.append(
                {
                    "id": b["id"],
                    "from": b["plan_center"],
                    "to": center,
                    "distance": round(math.dist(b["plan_center"], center), 3),
                }
            )
            b["plan_center"] = center
    report = {
        "algorithm": "bounded grid search with four deterministic allocation orders",
        "max_shift_relative_units": max_shift,
        "changes": changes,
        "unresolved_ids": best[2],
        "before": geometry_checks(original),
        "after": geometry_checks(output),
        "source_geometry_changed": False,
        "identities_changed": False,
        "sizes_or_facing_changed": False,
        "meaning": "Numerical layout regularization only; moved centers remain assumptions, not detected ground truth.",
    }
    output["layout_optimizations"] = output.get("layout_optimizations", []) + [report]
    return output, report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--max-shift", type=int, default=40)
    a = p.parse_args()
    if not 0 <= a.max_shift <= 50 or a.max_shift % 2:
        p.error("--max-shift must be even and between 0 and 50")
    updated, report = optimize(read(a.plan), a.max_shift)
    folder = stamp("whole_plan_fitted")
    updated["parent_plan"] = str(a.plan.resolve())
    save(folder / "plan.json", updated)
    save(folder / "fit_report.json", report)
    print(json.dumps({"plan": str(folder / "plan.json"), **report}, ensure_ascii=False), flush=True)
