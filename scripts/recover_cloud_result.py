"""Audited schema revalidation and explicitly bounded cloud corrections.

Raw request/result artifacts are immutable. Metadata-only normalization writes
a derived result. Missing geometry must be supplied by a new cloud response,
never invented by this helper. Network/truncation failures are not retried.
"""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil

from ancientplan.foundation2d.complete_plan import read, save, stamp, request
from ancientplan.foundation2d.plan_schema import normalize_plan_envelope, validate_plan_stage
from ancientplan.foundation2d.qwen_cloud import SafeError, model_matches


def recover_job(job, allow_cloud=False, max_cloud_corrections=1):
    if not job.get("exit_code"):
        return job
    origin = Path(job["run"])
    raw = read(origin / "result.json")
    meta = read(origin / "request_meta.json")
    stage = job["stage"]
    if (
        raw.get("finish_reason") != "stop"
        or not model_matches("qwen3.8-max", raw.get("model_returned"))
        or not isinstance(raw.get("parsed"), dict)
        or meta["stage"] != stage
    ):
        raise SafeError("Recovery refuses truncated, non-JSON or wrong-model responses")
    context = read(meta["context_source"]) if meta.get("context_source") else {}
    parsed, changes = normalize_plan_envelope(stage, raw["parsed"], context)
    issues = validate_plan_stage(stage, parsed, context)
    previous = {"run": str(origin), "validation_issues": raw["validation_issues"]}
    if not issues:
        folder = stamp(stage + "_schema_revalidated")
        for name in ("request_meta.json", "input.jpg", "answer.md"):
            if (origin / name).is_file():
                shutil.copyfile(origin / name, folder / name)
        result = deepcopy(raw)
        result["parsed_raw"] = raw.get("parsed_raw", raw["parsed"])
        result["parsed"] = parsed
        result["validation_issues"] = []
        result["metadata_normalizations"] = changes
        result["schema_revalidation"] = {
            "original_run": str(origin),
            "original_result_sha256": hashlib.sha256(
                (origin / "result.json").read_bytes()
            ).hexdigest(),
            "original_issues": raw["validation_issues"],
            "geometry_changed": False,
            "additional_cloud_calls": 0,
        }
        save(folder / "result.json", result)
        return {
            **job,
            "run": str(folder),
            "exit_code": 0,
            "previous_attempts": job.get("previous_attempts", []) + [previous],
        }
    if not allow_cloud:
        raise SafeError(
            "Substantive errors require explicit cloud correction: " + "; ".join(issues)
        )
    corrections = job.get("schema_correction_count", int(bool(job.get("schema_correction_from"))))
    if corrections >= max_cloud_corrections:
        raise SafeError("Schema correction budget exhausted; inspect the response")
    shape_diagnostics = []
    for collection, field in (
        ("objects", "footprint_image"),
        ("buildings", "source_footprint"),
        ("replace_buildings", "source_footprint"),
        ("add_buildings", "source_footprint"),
    ):
        objects = raw["parsed"].get(collection, [])
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            corners = obj.get(field)
            if isinstance(corners, list) and len(corners) != 4:
                shape_diagnostics.append(
                    {
                        "id": obj.get("id"),
                        "field": field,
                        "actual_points": len(corners),
                        "required_points": 4,
                        "note": "必须是四个点按环绕顺序形成凸四边形。不要输出五边形；裁切地基也须给有依据的四角近似并说明截断不确定性。",
                    }
                )
            size, interval = obj.get("plan_size"), obj.get("ratio_range")
            if (
                isinstance(size, list)
                and len(size) == 2
                and isinstance(interval, list)
                and len(interval) == 2
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in size + interval
                )
                and size[1] > 0
                and not interval[0] <= size[0] / size[1] <= interval[1]
            ):
                shape_diagnostics.append(
                    {
                        "id": obj.get("id"),
                        "field": "plan_size/ratio_range",
                        "actual_width_divided_by_depth": size[0] / size[1],
                        "declared_interval": interval,
                        "note": "实际前墙宽/进深必须位于有图像依据的声明区间内；不要单纯为消碰撞缩房子或编造比例。请复核图像尺寸并保持数值一致。",
                    }
                )
    context = {
        **context,
        "schema_correction": {
            "rejected_response": raw["parsed"],
            "validation_issues": issues,
            "shape_diagnostics": shape_diagnostics,
            "instruction": "上一份回答未通过格式/几何校验。请重新看原画，输出本阶段完整JSON，修正上述问题；不要只给patch。保留有证据的对象，不删房子来消除重叠，不编造坐标或边界。stage和全部必需字段必须存在。所有坐标均在0到1000内。",
        },
    }
    references = [
        {
            "image": r["image"]["source_path"],
            "crop": r["image"]["crop_in_source_pixels"],
            "label": r["label"],
        }
        for r in meta.get("reference_images", [])
    ]
    result = request(
        stage,
        meta["image"]["source_path"],
        crop=job.get("crop"),
        context=context,
        label=job.get("label"),
        references=references,
    )
    result["schema_correction_from"] = str(origin)
    result["schema_correction_count"] = corrections + 1
    result["previous_attempts"] = job.get("previous_attempts", []) + [previous]
    return result
