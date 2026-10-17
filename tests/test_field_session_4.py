"""Round four of field findings (issue #45 log 8 and issue #53, 2026-10-16).

The dog heard its own one-word "D'accord !" six times in a row (short phrases
were never checked against its speech), the reporter could not tell WHY his
brain's LLM failed (the reason sat in a journal he had no reason to read), and
"/mcp says connected" did not tell him whether Claude could reach the robot.
"""
import importlib.util
import io
import json
import os
import socket
import sys
import urllib.error

import pytest

import body.nox_voice_loop_v2 as v2
import brain.nox_mcp_server as mcp
import brain.nox_voice_brain as vb


def state(recent=(), speaking=False):
    return {"ok": True, "speaking": speaking, "recent": list(recent)}


# ─── the "d'accord" loop ───

def test_the_dogs_one_word_reply_heard_back_is_an_echo():
    # Log 8, 11:06:34-11:07:36: "Heard: 'd'accord'" six times, each one the
    # dog's own "D'accord !" fed back in as a new sentence.
    s = state([{"text": "D'accord !", "age_s": 3.2}])
    assert v2.is_own_speech("d'accord", s) is True


def test_a_short_command_is_never_eaten_by_a_longer_sentence():
    # The dog said a sentence containing "assis"; the user saying "assis"
    # afterwards is a real command, not an echo.
    s = state([{"text": "Je suis assis maintenant.", "age_s": 2.0}])
    assert v2.is_own_speech("assis", s) is False


def test_short_echo_only_inside_the_window():
    s = state([{"text": "D'accord !", "age_s": v2.ECHO_TEXT_WINDOW + 1}])
    assert v2.is_own_speech("d'accord", s) is False


def test_different_short_phrase_is_not_an_echo():
    s = state([{"text": "D'accord !", "age_s": 2.0}])
    assert v2.is_own_speech("assis", s) is False


def test_long_phrase_overlap_still_works():
    s = state([{"text": "Rien de particulier pour le moment, tu peux commencer "
                        "par me présenter toi-même.", "age_s": 4.0}])
    heard = "rien de particulier pour le moment tu peux commencer par me présenter toi-même"
    assert v2.is_own_speech(heard, s) is True


# ─── every LLM failure names its fix ───

REMOTE = "http://192.0.2.1:11434/v1/chat/completions"
LOCAL = "http://127.0.0.1:11434/v1/chat/completions"


def refused():
    return urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


def test_refused_remote_ollama_explains_the_systemd_way():
    hint = vb.llm_error_hint(refused(), REMOTE)
    assert "OLLAMA_HOST=0.0.0.0" in hint and "systemctl edit ollama" in hint
    assert "export" in hint  # the shell-export trap, named


def test_refused_local_endpoint_asks_whether_ollama_runs():
    hint = vb.llm_error_hint(refused(), LOCAL)
    assert "systemctl status ollama" in hint and "OLLAMA_HOST" not in hint


def test_refused_other_endpoint_blames_address_or_firewall():
    hint = vb.llm_error_hint(refused(), "http://192.0.2.1:8080/v1/chat/completions")
    assert "firewall" in hint


def test_missing_model_gets_the_pull_command():
    err = urllib.error.HTTPError(REMOTE, 404, "Not Found", {}, io.BytesIO(b""))
    hint = vb.llm_error_hint(err, REMOTE, "llama3.2")
    assert "ollama pull llama3.2" in hint


def test_rejected_key_points_at_the_key():
    err = urllib.error.HTTPError(REMOTE, 401, "Unauthorized", {}, io.BytesIO(b""))
    assert "OPENAI_API_KEY" in vb.llm_error_hint(err, REMOTE)


def test_unknown_host_name_suggests_the_ip():
    err = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    assert "IP address" in vb.llm_error_hint(err, "http://shyrka:11434/v1/chat/completions")


def test_timeouts_are_recognised():
    assert vb.is_timeout(urllib.error.URLError(socket.timeout("timed out"))) is True
    assert vb.is_timeout(refused()) is False


def test_call_llm_logs_the_hint_and_returns_none(monkeypatch, capsys):
    def boom(*a, **k):
        raise refused()
    monkeypatch.setattr(vb.urllib.request, "urlopen", boom)
    monkeypatch.setattr(vb, "OPENAI_URL", REMOTE)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    assert vb.call_llm([{"role": "user", "content": "assis"}]) is None
    out = capsys.readouterr().out
    assert "LLM API error" in out and "OLLAMA_HOST=0.0.0.0" in out


# ─── scripts/llm_check.py ───

def load_llm_check():
    spec = importlib.util.spec_from_file_location("llm_check", "scripts/llm_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, content):
        self._body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def check(monkeypatch, tmp_path):
    """llm_check.main with a fresh brain import, the env only via monkeypatch."""
    lc = load_llm_check()
    monkeypatch.setattr(lc, "brain_service_state", lambda: "unknown")
    monkeypatch.setattr(lc, "probe_port", lambda url, timeout=5.0: None)
    for key in ("OPENAI_URL", "OPENAI_API_KEY", "LLM_MODEL", "NOX_MODE", "NOX_LANG",
                "LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "nox-brain"
    env_file.write_text("# test\n")

    def run(env, urlopen, probe=None):
        if probe is not None:
            monkeypatch.setattr(lc, "probe_port", lambda url, timeout=5.0: probe)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr("urllib.request.urlopen", urlopen)
        sys.modules.pop("nox_voice_brain", None)
        try:
            return lc.main(["llm_check.py", str(env_file)])
        finally:
            sys.modules.pop("nox_voice_brain", None)
    return run


def test_llm_check_reports_the_answer_and_its_time(check, capsys):
    reply = '{"speak": "D\'accord !", "actions": ["sit", "wag_tail"]}'
    code = check({"OPENAI_URL": LOCAL, "LLM_MODEL": "llama3.2", "NOX_LANG": "fr"},
                 lambda *a, **k: FakeResponse(reply))
    out = capsys.readouterr().out
    assert code == 0
    assert "Assieds-toi" in out and "Answer in" in out
    assert "['sit', 'wag_tail']" in out


def test_llm_check_names_the_fix_when_refused(check, capsys):
    def boom(*a, **k):
        raise refused()
    code = check({"OPENAI_URL": REMOTE, "LLM_MODEL": "llama3.2"}, boom)
    out = capsys.readouterr().out
    assert code == 1 and "FAILED" in out and "OLLAMA_HOST=0.0.0.0" in out


def test_llm_check_tells_a_silent_host_from_a_slow_model(check, capsys):
    # A firewall that DROPs packets looks like a timeout to HTTP — the tool
    # must not blame the model for it.
    code = check({"OPENAI_URL": REMOTE}, lambda *a, **k: pytest.fail("no request"),
                 probe=socket.timeout("timed out"))
    out = capsys.readouterr().out
    assert code == 1 and "does not answer at all" in out and "LLM_TIMEOUT" not in out


def test_llm_check_refused_port_gets_the_hint_before_any_request(check, capsys):
    code = check({"OPENAI_URL": REMOTE}, lambda *a, **k: pytest.fail("no request"),
                 probe=ConnectionRefusedError(111, "Connection refused"))
    assert code == 1 and "OLLAMA_HOST=0.0.0.0" in capsys.readouterr().out


def test_probe_port_reports_a_closed_port():
    lc = load_llm_check()
    with socket.socket() as s:  # grab a free port, then close it again
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert isinstance(lc.probe_port(f"http://127.0.0.1:{port}/v1"), OSError)


def test_llm_check_warns_when_the_model_ignores_json(check, capsys):
    code = check({"OPENAI_URL": LOCAL}, lambda *a, **k: FakeResponse("Ouaf ! Je m'assieds."))
    assert code == 0 and "without the JSON" in capsys.readouterr().out


def test_llm_check_says_when_nothing_is_configured(check, capsys):
    code = check({}, lambda *a, **k: pytest.fail("must not call the LLM"))
    assert code == 1 and "No LLM configured" in capsys.readouterr().out


def test_env_file_reader_lets_the_environment_win(tmp_path):
    lc = load_llm_check()
    f = tmp_path / "env"
    f.write_text('# comment\nOPENAI_URL="http://x:1/v1"\nLLM_MODEL=llama3.2\nbroken line\n')
    assert lc.read_env_file(str(f), {"LLM_MODEL": "mine"}) == {"OPENAI_URL": "http://x:1/v1"}


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root reads everything")
def test_root_only_env_file_says_use_sudo(tmp_path, capsys):
    lc = load_llm_check()
    f = tmp_path / "nox-brain"
    f.write_text("OPENAI_URL=x\n")
    f.chmod(0)
    try:
        assert lc.main(["llm_check.py", str(f)]) == 1
        assert "sudo python3 scripts/llm_check.py" in capsys.readouterr().out
    finally:
        f.chmod(0o600)


# ─── nox_mcp_server.py --check ───

def test_mcp_check_succeeds_when_the_robot_answers(monkeypatch, capsys):
    monkeypatch.setattr(mcp, "bridge", lambda path, timeout=30: {"ok": True, "battery": 7.4})
    assert mcp.check() == 0
    assert "Esc" in capsys.readouterr().out


def test_mcp_check_shows_why_the_robot_is_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(mcp, "bridge", lambda path, timeout=30: {
        "ok": False, "error": "robot not reachable at http://pidog.local:8888 (refused)"})
    assert mcp.check() == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "pidog.local" in out


# ─── an empty answer is a failure with a reason (issue #45, log 9: qwen3-VL) ───

def test_empty_llm_answer_is_logged_and_treated_as_failure(monkeypatch, capsys):
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    monkeypatch.setattr(vb, "request_llm", lambda *a, **k: "  ")
    assert vb.call_llm([{"role": "user", "content": "assis"}]) is None
    out = capsys.readouterr().out
    assert "empty answer" in out and "reasoning models" in out


def test_llm_check_reports_an_empty_answer_as_failure(check, capsys):
    # Log 9: qwen3-VL took 55 s and returned "", and the tool claimed the dog
    # "will speak this".
    code = check({"OPENAI_URL": LOCAL, "LLM_MODEL": "qwen3-VL"},
                 lambda *a, **k: FakeResponse(""))
    out = capsys.readouterr().out
    assert code == 1 and "NOTHING" in out and "will speak this" not in out
