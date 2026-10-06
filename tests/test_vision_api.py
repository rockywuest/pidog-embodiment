"""/vision after the models were downloaded — issue #36.

The reporter built llama.cpp, downloaded both models and got
"vision not running (no result file)": nothing had started the nox-vision
service, and nothing said so. These tests pin what /vision tells the user in
each state.
"""
import json
import time
import urllib.error
import urllib.request

import body.nox_brain_bridge as bridge
from tests.test_bridge_api import server_port  # noqa: F401 - fixture


def get_vision(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/vision", timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_no_result_file_says_how_to_start_the_service(server_port, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(bridge, "VISION_RESULT_FILE", str(tmp_path / "missing.json"))
    status, body = get_vision(server_port)
    assert status == 503 and body["ok"] is False
    assert "install-body.sh nox-vision" in body["error"]
    assert "journalctl -u nox-vision" in body["error"]


def test_a_failed_setup_is_reported_with_its_reason(server_port, monkeypatch, tmp_path):  # noqa: F811
    f = tmp_path / "vision.json"
    f.write_text(json.dumps({"ts": time.time(), "description": None, "prompt_type": "setup",
                             "error": "projector not found: /home/x/models/smolvlm/mmproj.gguf"}))
    monkeypatch.setattr(bridge, "VISION_RESULT_FILE", str(f))
    status, body = get_vision(server_port)
    assert body["ok"] is False
    assert "projector not found" in body["error"]


def test_a_scene_description_is_ok(server_port, monkeypatch, tmp_path):  # noqa: F811
    f = tmp_path / "vision.json"
    f.write_text(json.dumps({"ts": time.time() - 5, "description": "A cat on a sofa.",
                             "prompt_type": "describe", "error": None}))
    monkeypatch.setattr(bridge, "VISION_RESULT_FILE", str(f))
    status, body = get_vision(server_port)
    assert status == 200 and body["ok"] is True
    assert body["description"] == "A cat on a sofa." and body["age_s"] >= 5
