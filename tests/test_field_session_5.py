"""Round five of field findings (issue #45, log 10, 2026-10-17).

Vosk heard the dog's own "D'accord !" as 'dakar' — an echo no word match can
catch; the reporter suggested a plain "OK", which the voice loop already drops
as filler. His brain and Ollama ran on the same machine, yet OPENAI_URL used
the LAN address where Ollama does not listen. And every voice log opened with
'daemon error: Connection refused' because the voice loop spoke to the daemon
before it was up.
"""
import importlib.util
import socket


import body.nox_voice_loop_v2 as v2
import brain.nox_voice_brain as vb


# ─── the French acknowledgement can't turn into 'dakar' ───

def test_french_ok_reply_is_plain_ok():
    assert vb._REPLIES["ok"]["fr"] == "OK !"


def test_ok_heard_back_is_dropped_as_filler():
    assert v2.is_just_noise("ok") and v2.is_just_noise("okay")


def test_no_d_accord_left_in_the_prompt_examples():
    # small models copy the examples word for word
    assert '"speak":"D\'accord' not in vb.SYSTEM_PROMPT


# ─── llm_check: Ollama on this machine, URL on the LAN address ───

def load_llm_check():
    spec = importlib.util.spec_from_file_location("llm_check", "scripts/llm_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loopback_twin_of_a_lan_url():
    lc = load_llm_check()
    assert (lc.same_port_on_loopback("http://192.0.2.7:11434/v1/chat/completions")
            == "http://127.0.0.1:11434/v1/chat/completions")
    assert lc.same_port_on_loopback("http://127.0.0.1:11434/v1") is None
    assert lc.same_port_on_loopback("https://api.openai.com/v1/chat/completions") is None


def test_llm_check_says_use_loopback_when_ollama_runs_here(monkeypatch, tmp_path, capsys):
    import sys
    lc = load_llm_check()
    monkeypatch.setattr(lc, "brain_service_state", lambda: "unknown")
    monkeypatch.setattr(lc, "probe_port", lambda url, timeout=5.0: (
        None if "127.0.0.1" in url else ConnectionRefusedError(111, "Connection refused")))
    for key in ("OPENAI_API_KEY", "LLM_MODEL", "NOX_MODE", "NOX_LANG", "LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_URL", "http://192.0.2.7:11434/v1/chat/completions")
    env_file = tmp_path / "nox-brain"
    env_file.write_text("# test\n")
    sys.modules.pop("nox_voice_brain", None)
    try:
        code = lc.main(["llm_check.py", str(env_file)])
    finally:
        sys.modules.pop("nox_voice_brain", None)
    out = capsys.readouterr().out
    assert code == 1
    assert "OPENAI_URL=http://127.0.0.1:11434/v1/chat/completions" in out
    assert "systemctl edit ollama" not in out   # not the remote-machine advice


# ─── the voice loop waits for the daemon instead of logging errors ───

def test_wait_for_daemon_returns_once_the_port_answers(monkeypatch):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    monkeypatch.setattr(v2, "DAEMON_HOST", "127.0.0.1")
    monkeypatch.setattr(v2, "DAEMON_PORT", srv.getsockname()[1])
    try:
        assert v2.wait_for_daemon(max_wait=2.0) is True
    finally:
        srv.close()


def test_wait_for_daemon_gives_up_quietly(monkeypatch, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(v2, "DAEMON_HOST", "127.0.0.1")
    monkeypatch.setattr(v2, "DAEMON_PORT", port)
    assert v2.wait_for_daemon(max_wait=0.3, interval=0.1) is False
    assert "daemon error" not in capsys.readouterr().out
