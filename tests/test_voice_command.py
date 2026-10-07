"""Voice commands without an LLM, and whether the brain got them — issue #42.

The reporter sent the README's own example, "Setz dich hin und wedel mit dem
Schwanz!", to /voice/input and the dog only said "Ich habe verstanden: ...":
the keyword fallback knew "sitz" but not "setz", had no "wedel", ran only one
command per sentence, matched "go" inside "good dog", and spoke no French.
/voice/input also answered ok:true while the push to the brain failed, and the
brain never read the pushes it received.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import body.nox_brain_bridge as bridge
import brain.nox_voice_brain as vb
from tests.test_bridge_api import server_port  # noqa: F401 - fixture


@pytest.mark.parametrize("text,actions,lang", [
    ("Setz dich hin und wedel mit dem Schwanz!", ["sit", "wag_tail"], "de"),
    ("Platz!", ["lie"], "de"),
    ("Komm her!", ["forward"], "de"),
    ("Assis-toi", ["sit"], "fr"),
    ("Assieds-toi et remue la queue", ["sit", "wag_tail"], "fr"),
    ("Couché !", ["lie"], "fr"),
    ("Lève-toi", ["stand"], "fr"),
    ("Aboie !", ["bark"], "fr"),
    ("sit down and wag your tail", ["sit", "wag_tail"], "en"),
    ("Down!", ["lie"], "en"),
    ("come here and turn left", ["forward", "turn_left"], "en"),
])
def test_commands_in_three_languages(text, actions, lang):
    cmd = vb.parse_simple_command(text)
    assert cmd["actions"] == actions and cmd["intent"] == "command" and cmd["lang"] == lang


@pytest.mark.parametrize("text,intent,lang", [
    ("Good dog", "praise", "en"),     # used to walk forward: "go" in "good"
    ("Merci", "praise", "fr"),
    ("Brav!", "praise", "de"),
    ("Wer bist du?", "who", "de"),
])
def test_praise_and_identity_wag_instead_of_walking(text, intent, lang):
    cmd = vb.parse_simple_command(text)
    assert cmd["intent"] == intent and cmd["actions"] == ["wag_tail"] and cmd["lang"] == lang


@pytest.mark.parametrize("text,lang", [("Was ist das?", "de"), ("Qu'est-ce que tu vois ?", "fr"),
                                       ("What is that?", "en")])
def test_unknown_sentences_are_answered_in_their_language(text, lang, monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: sent.append((path, data)))
    vb.process_voice_simple({"text": text})
    assert sent[0][0] == "/speak"
    assert sent[0][1]["text"] == vb._REPLIES["unknown"][lang].format(text=text)


def test_a_command_becomes_one_combo(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: sent.append((path, data)))
    vb.process_voice_simple({"text": "Setz dich hin und wedel mit dem Schwanz!"})
    assert sent == [("/combo", {"actions": ["sit", "wag_tail"], "speak": "Mach ich!"})]


def test_pushed_messages_are_processed_not_just_queued():
    handled = []
    vb._push_queue.put({"text": "Platz"})
    vb.wait_for_push(handled.append, 0.2)
    assert handled == [{"text": "Platz"}]


# ─── bridge: /voice/input tells the caller whether a brain got it ───

class _Brain(BaseHTTPRequestHandler):
    reply = b'{"ok":true,"processes":true}'
    received = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).received.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(type(self).reply)


@pytest.fixture
def fake_brain(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Brain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Brain.received = []
    monkeypatch.setattr(bridge, "BRAIN_HOST", "127.0.0.1")
    monkeypatch.setattr(bridge, "BRAIN_CALLBACK_PORT", srv.server_address[1])
    with bridge.perception.lock:
        bridge.perception.voice_inbox.clear()
    yield _Brain
    srv.shutdown()


def post_voice(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/voice/input",
                                 data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_brain_that_processes_pushes_gets_it_exactly_once(server_port, fake_brain):  # noqa: F811
    body = post_voice(server_port, "Platz")
    assert body == {"ok": True, "brain": "received"}
    assert fake_brain.received[0]["text"] == "Platz"
    assert bridge.perception.voice_inbox == []  # not a second time via polling


def test_an_older_brain_still_gets_it_through_the_inbox(server_port, fake_brain, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fake_brain, "reply", b'{"ok":true}')
    assert post_voice(server_port, "Platz")["ok"] is True
    assert [m["text"] for m in bridge.perception.voice_inbox] == ["Platz"]


def test_no_brain_is_reported_and_the_message_waits(server_port, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bridge, "BRAIN_HOST", "127.0.0.1")
    monkeypatch.setattr(bridge, "BRAIN_CALLBACK_PORT", 9)  # nothing listens
    with bridge.perception.lock:
        bridge.perception.voice_inbox.clear()
    body = post_voice(server_port, "Platz")
    assert body["ok"] is False and body["queued"] is True
    assert "brain not reachable at 127.0.0.1:9" in body["error"]
    assert [m["text"] for m in bridge.perception.voice_inbox] == ["Platz"]
