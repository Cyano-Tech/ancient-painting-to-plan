"""A new review is explicit and cannot overwrite the previous plan snapshot."""

from scripts.presentation_batch import next_review
from ancientplan.foundation2d.complete_plan import read, save


def test_next_review_preserves_old_plan_and_review_history(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    plan = scene / "plan.json"
    save(plan, {"source": {"sha256": "source_digest"}})
    run = scene / "run"
    run.mkdir()
    save(
        run / "result.json",
        {
            "model_returned": "qwen3.8-max",
            "finish_reason": "stop",
            "validation_issues": [],
            "parsed": {"verdict": "needs_revision", "blocking_issues": [{"ids": ["H1"]}]},
        },
    )
    save(
        scene / "state.json",
        {
            "reviewed_plan": str(plan),
            "final_check": {"run": str(run)},
            "ready": False,
            "critique": {"run": "previous"},
        },
    )
    before = plan.read_bytes()
    next_review(
        tmp_path, {"directory": "scene"}, ["Inspect support chain"], independent_identity=True
    )
    state = read(scene / "state.json")
    assert state["fitted_plan"] == str(plan)
    assert "reviewed_plan" not in state and "final_check" not in state
    assert state["review_history"][0]["final_check"]["run"] == str(run)
    assert state["independent_identity_requested"] and not state["ready"]
    assert read(state["review_notes"])["source_sha256"] == "source_digest"
    assert plan.read_bytes() == before
