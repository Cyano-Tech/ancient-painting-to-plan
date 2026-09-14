"""Exercise the cloud orchestration with synthetic responses; never send requests."""

import argparse
import json
from pathlib import Path

from PIL import Image
import pytest

from ancientplan.foundation2d import qwen_cloud as client


@pytest.fixture
def fake_request(tmp_path, monkeypatch):
    key_file = tmp_path / "fake.env"
    key_file.write_text("DASHSCOPE_API_KEY='synthetic-test-credential'\n")
    key_file.chmod(0o600)
    image = tmp_path / "input.png"
    Image.new("RGB", (24, 16), "white").save(image)
    config = {
        "base_url": "https://example.invalid/v1",
        "model": "qwen3.8-max",
        "cloud_only": True,
        "local_model_fallback": False,
        "api_key_file": str(key_file),
        "request_timeout_seconds": 1,
        "smoke": {"max_image_edge": 24, "max_tokens": 20, "enable_thinking": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("ANCIENTPLAN_CONFIG", str(path))
    monkeypatch.setattr(client, "RUN_ROOT", tmp_path / "runs")
    args = argparse.Namespace(
        stage="smoke",
        image=str(image),
        crop=None,
        base_url=None,
        run_label=None,
        context=None,
        reference_images=[],
    )
    return args


@pytest.mark.parametrize(
    "returned,finish,expected",
    [
        ("qwen3.8-max", "stop", 0),
        ("qwen3.8-flash", "stop", 2),
        ("qwen3.8-max", "length", 2),
    ],
)
def test_response_gate_and_credential_redaction(
    fake_request, monkeypatch, returned, finish, expected
):
    calls = []

    def transport(url, key, body, timeout):
        calls.append((url, body["model"]))
        return {
            "model": returned,
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": '{"text":"synthetic-test-credential"}'},
                }
            ],
        }

    monkeypatch.setattr(client, "send_request", transport)
    assert client.run(fake_request) == expected
    assert calls == [("https://example.invalid/v1", "qwen3.8-max")]
    folder = Path(fake_request.output_dir)
    for name in ("request_meta.json", "result.json", "answer.md"):
        assert "synthetic-test-credential" not in (folder / name).read_text()
    assert json.loads((folder / "request_meta.json").read_text())["gpu_used"] is False


def test_missing_config_fails_before_reading_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANCIENTPLAN_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(client.SafeError, match="config.example"):
        client.run(argparse.Namespace())


def test_redirects_are_refused():
    with pytest.raises(client.SafeError, match="redirect refused"):
        client.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid")
