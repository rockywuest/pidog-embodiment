"""The brain against non-OpenAI endpoints (Ollama & friends) — 2026-10-09 sweep.

Two failure modes with local models: the endpoint rejects OpenAI's
response_format with HTTP 400 (the whole LLM path then looked dead), and a
small model answers in plain text instead of JSON (the dog then recited
"sit down" instead of sitting).
"""
import io
import json
import urllib.error

import pytest

import brain.nox_voice_brain as vb


@pytest.fixture(autouse=True)
def _fresh_rejected_params(monkeypatch):
    # request_llm remembers refused parameters for the process; tests must not inherit them
    monkeypatch.setattr(vb, "_REJECTED_PARAMS", set())


def http400(url):
    return urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"response_format"))


def test_llm_retries_without_response_format_on_400(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        if "response_format" in payload:
            raise http400(req.full_url)

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"speak":"Ok!","actions":["sit"]}'}}]}).encode()
        return R()

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    out = vb.call_llm([{"role": "user", "content": "sit"}])
    assert out == '{"speak":"Ok!","actions":["sit"]}'
    assert len(calls) == 2 and "response_format" not in calls[1]


def test_other_llm_errors_are_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b""))

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    assert vb.call_llm([{"role": "user", "content": "sit"}]) is None
    assert len(calls) == 1


def test_plain_text_reply_still_executes_the_users_command(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_get", lambda path, timeout=10: {"error": "offline"})
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: sent.append((path, data)))
    monkeypatch.setattr(vb, "call_llm", lambda messages: "Okay, I will sit down now, friend!")
    vb.process_voice_intelligent({"text": "Setz dich hin und wedel mit dem Schwanz!"})
    combos = [d for p, d in sent if p == "/combo"]
    assert combos and combos[0]["actions"] == ["sit", "wag_tail"]
    assert combos[0]["speak"] == "Okay, I will sit down now, friend!"


def test_a_json_reply_with_no_actions_is_respected(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_get", lambda path, timeout=10: {"error": "offline"})
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: sent.append((path, data)))
    monkeypatch.setattr(vb, "call_llm", lambda messages: '{"speak":"Ich bin ein Hund.","actions":[]}')
    vb.process_voice_intelligent({"text": "Wer bist du? Sitz ist kein Befehl hier."})
    combos = [d for p, d in sent if p == "/combo"]
    assert combos and "actions" not in combos[0]  # deliberate empty → no keyword override


def test_push_port_follows_brain_callback_port(monkeypatch):
    import importlib
    monkeypatch.setenv("BRAIN_CALLBACK_PORT", "18890")
    importlib.reload(vb)
    try:
        assert vb.PUSH_PORT == 18890
    finally:
        monkeypatch.delenv("BRAIN_CALLBACK_PORT")
        importlib.reload(vb)


def _ok_response():
    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"speak":"Ok!","actions":["sit"]}'}}]}).encode()
    return R()


def test_claude_temperature_400_is_dropped_and_remembered(monkeypatch):
    # Opus 5.5 via Anthropic's OpenAI-compatible endpoint, verbatim error
    calls = []
    body = b'{"error": {"message": "`temperature` is deprecated for this model."}}'

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        if "temperature" in payload:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(body))
        return _ok_response()

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    monkeypatch.setattr(vb, "LLM_TEMPERATURE", 0.7)
    monkeypatch.setattr(vb, "_REJECTED_PARAMS", set())
    assert vb.call_llm([{"role": "user", "content": "sit"}]) == '{"speak":"Ok!","actions":["sit"]}'
    # response_format survives — only what the error named is dropped
    assert len(calls) == 2 and "temperature" not in calls[1] and "response_format" in calls[1]
    # the next sentence goes out without temperature right away
    vb.call_llm([{"role": "user", "content": "stand"}])
    assert len(calls) == 3 and "temperature" not in calls[2]


def test_temperature_off_sends_none(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _ok_response()

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    monkeypatch.setattr(vb, "LLM_TEMPERATURE", None)
    monkeypatch.setattr(vb, "_REJECTED_PARAMS", set())
    vb.call_llm([{"role": "user", "content": "sit"}])
    assert "temperature" not in calls[0]


def test_a_400_naming_nothing_droppable_is_not_retried_forever(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"model not found"))

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    monkeypatch.setattr(vb, "_REJECTED_PARAMS", set())
    assert vb.call_llm([{"role": "user", "content": "sit"}]) is None
    assert len(calls) == 2  # one retry without response_format, then give up


def test_both_params_refused_in_turn_still_answers_the_first_sentence(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        for key in ("temperature", "response_format"):
            if key in payload:
                raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                             io.BytesIO(f"`{key}` is not supported".encode()))
        return _ok_response()

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    monkeypatch.setattr(vb, "LLM_TEMPERATURE", 0.7)
    assert vb.call_llm([{"role": "user", "content": "sit"}]) == '{"speak":"Ok!","actions":["sit"]}'
    assert len(calls) == 3 and vb._REJECTED_PARAMS == {"temperature", "response_format"}


def test_an_unknown_400_shows_what_the_endpoint_said(monkeypatch, capsys):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"error": "model claude-opus-9 not found"}'))

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    assert vb.call_llm([{"role": "user", "content": "sit"}]) is None
    assert "claude-opus-9 not found" in capsys.readouterr().out


def test_an_answer_cut_by_the_budget_says_raise_llm_max_tokens(monkeypatch, capsys):
    def fake_urlopen(req, timeout=None):
        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": ""},
                                                "finish_reason": "length"}]}).encode()
        return R()

    monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vb, "LLM_CONFIGURED", True)
    assert vb.call_llm([{"role": "user", "content": "sit"}]) is None
    out = capsys.readouterr().out
    assert "LLM_MAX_TOKENS" in out and "llama3.2" not in out


@pytest.mark.parametrize("raw, value", [("0.7", 0.7), ("off", None), ("", None), (" OFF ", None), ("0", 0.0)])
def test_temperature_parsing(raw, value):
    assert vb._parse_temperature(raw) == value


@pytest.mark.parametrize("raw", ["abc", "0,7", "nan", "inf"])
def test_a_bad_temperature_names_the_setting(raw):
    with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
        vb._parse_temperature(raw)
