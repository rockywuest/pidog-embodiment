"""The brain against non-OpenAI endpoints (Ollama & friends) — 2026-10-09 sweep.

Two failure modes with local models: the endpoint rejects OpenAI's
response_format with HTTP 400 (the whole LLM path then looked dead), and a
small model answers in plain text instead of JSON (the dog then recited
"sit down" instead of sitting).
"""
import io
import json
import urllib.error

import brain.nox_voice_brain as vb


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
