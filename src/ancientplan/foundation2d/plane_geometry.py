"""Small CPU-only projective maps for explicitly assumed ground quadrilaterals.

Mapping a quad to a rectangle does not prove camera calibration or true shape.
All units are relative. These helpers never infer corners or aspect ratios.
"""

import math


def solve(matrix, rhs):
    n = len(rhs)
    aug = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Degenerate projective constraints")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [v - factor * w for v, w in zip(aug[row], aug[col])]
    return [row[-1] for row in aug]


def homography(source, target):
    if len(source) != 4 or len(target) != 4:
        raise ValueError("Four correspondences required")
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(source, target):
        rows.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        rhs.extend([u, v])
    return solve(rows, rhs) + [1.0]


def transform(matrix, xy):
    x, y = xy
    denominator = matrix[6] * x + matrix[7] * y + matrix[8]
    if abs(denominator) < 1e-12:
        raise ValueError("Point lies on projective horizon")
    result = [
        (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
        (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
    ]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Non-finite projection")
    return result


def plane_mapping(corners, width_depth_ratio):
    if not math.isfinite(width_depth_ratio) or width_depth_ratio <= 0:
        raise ValueError("Positive finite aspect ratio required")
    src = [[x / 1000, y / 1000] for x, y in corners]
    dst = [[0, 0], [width_depth_ratio, 0], [width_depth_ratio, 1], [0, 1]]
    forward = homography(src, dst)
    inverse = homography(dst, src)
    error = max(math.dist(transform(forward, p), q) for p, q in zip(src, dst))
    return {
        "image_to_plane": forward,
        "plane_to_image": inverse,
        "relative_width": width_depth_ratio,
        "relative_depth": 1,
        "corner_fit_error": error,
        "calibration_verified": False,
        "note": "Rectangle and metric ratio were assumed; fit error is not visual accuracy.",
    }


def hypothesis_grid(corners, ratio, divisions=4):
    mapping = plane_mapping(corners, ratio)
    inv = mapping["plane_to_image"]

    def image_point(p):
        return [v * 1000 for v in transform(inv, p)]

    lines = []
    for i in range(1, divisions):
        v = i / divisions
        lines.append([image_point([0, v]), image_point([ratio, v])])
    count = min(120, math.ceil(ratio * divisions))
    for i in range(1, count):
        u = i / divisions
        if u < ratio:
            lines.append([image_point([u, 0]), image_point([u, 1])])
    return lines, mapping
