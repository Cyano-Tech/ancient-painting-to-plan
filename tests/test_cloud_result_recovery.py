"""Rejected responses remain traceable; format recovery never guesses geometry."""

import json

import pytest

from ancientplan.foundation2d.qwen_cloud import SafeError
from scripts import recover_cloud_result as recovery


def failed_job(tmp_path, parsed, issues, finish_reason="stop"):
    folder = tmp_path / "original"
    folder.mkdir()
    result = {
        "parsed": parsed,
        "validation_issues": issues,
        "finish_reason": finish_reason,
        "model_returned": "qwen3.8-max",
        "answer": json.dumps(parsed),
        "vision_accuracy_verified": False,
        "coverage_verified": False,
    }
    recovery.save(folder / "result.json", result)
    recovery.save(folder / "request_meta.json", {"stage": "plan_buildings"})
    return {
        "run": str(folder),
        "stage": "plan_buildings",
        "exit_code": 2,
        "label": "nw",
        "crop": None,
    }


def test_recovery_preserves_raw_and_records_derivation(tmp_path, monkeypatch):
    raw = {"objects": [], "excluded": [], "coverage_notes": "empty crop"}
    job = failed_job(tmp_path, raw, ["wrong stage"])
    original = (tmp_path / "original/result.json").read_bytes()
    target = tmp_path / "derived"
    target.mkdir()
    monkeypatch.setattr(recovery, "stamp", lambda _: target)
    monkeypatch.setattr(recovery, "request", lambda *a, **kw: pytest.fail("No cloud call allowed"))
    recovered = recovery.recover_job(job)
    assert recovered["exit_code"] == 0 and job["exit_code"] == 2
    result = recovery.read(target / "result.json")
    assert result["parsed"] == {**raw, "stage": "plan_buildings"}
    assert result["parsed_raw"] == raw
    assert not result["schema_revalidation"]["geometry_changed"]
    assert not result["vision_accuracy_verified"]
    assert (tmp_path / "original/result.json").read_bytes() == original


def test_missing_substantive_fields_do_not_get_filled_locally(tmp_path):
    job = failed_job(tmp_path, {"stage": "plan_buildings", "objects": [{}]}, ["invalid"])
    with pytest.raises(SafeError, match="Substantive errors"):
        recovery.recover_job(job)


def test_truncated_response_refused_even_when_correction_allowed(tmp_path):
    job = failed_job(tmp_path, {}, ["truncated"], finish_reason="length")
    with pytest.raises(SafeError, match="truncated"):
        recovery.recover_job(job, allow_cloud=True)


def test_exhausted_correction_budget_never_sends_another_request(tmp_path, monkeypatch):
    job = failed_job(tmp_path, {"stage": "plan_buildings", "objects": [{}]}, ["invalid"])
    job["schema_correction_count"] = 2
    monkeypatch.setattr(recovery, "request", lambda *a, **kw: pytest.fail("Budget exceeded"))
    with pytest.raises(SafeError, match="budget exhausted"):
        recovery.recover_job(job, allow_cloud=True, max_cloud_corrections=2)
