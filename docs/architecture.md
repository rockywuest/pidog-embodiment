# Architecture

*The current system design. For the visual diagram see the README; for the
original research notes see [research.md](research.md).*

## The core idea: Brain / Body split over HTTP

The intelligence does not live on the robot. A **brain** process (any machine —
Pi 5, desktop, laptop) talks to a **body** (the robot's Pi) over plain HTTP.
The body exposes sensors and actuators as an API; the brain decides what to do.

```
┌──────────────────────────────┐         ┌──────────────────────────────────┐
│  BRAIN (any machine)         │  HTTP   │  BODY (robot's Raspberry Pi)     │
│                              │ :8888   │                                  │
│  nox-brain ──────────────────┼────────►│  nox-bridge (HTTP API)           │
│  · LLM loop (OpenAI/Ollama)  │         │      │ TCP :9999 / unix socket   │
│  · decides actions/speech    │         │      ▼                           │
│  · telegram_bot (optional)   │         │  nox-body (nox_daemon.py)        │
│                              │         │  · SunFounder SDK (servos, IMU)  │
│  nox_body_client.py          │         │  · camera, RGB, TTS playback     │
│  · Python API + CLI          │         │  nox-voice (optional, Vosk STT)  │
│                              │         │  nox-vision (optional, SmolVLM)  │
└──────────────────────────────┘         └──────────────────────────────────┘
```

Both halves can also run on one machine (`BRAIN_HOST=127.0.0.1`).

## Components

| Component | File | Runs on | Role |
|-----------|------|---------|------|
| Daemon | `body/nox_daemon.py` | body | Owns the SunFounder SDK: servos, sensors, TTS, camera. Command server on TCP :9999 + unix socket. |
| Bridge | `body/nox_brain_bridge.py` | body | HTTP API on :8888. Translates REST calls into daemon commands, holds perception state. |
| Behavior engine | `body/nox_behavior_engine.py` | body | Local autonomy: mood FSM, idle behaviors, patrol — keeps the dog alive when the brain is away. |
| Voice loop | `body/nox_voice_loop_v2.py` | body | Optional: Vosk STT → forwards recognized text to the brain. Exits cleanly if no model. |
| Vision | `body/nox_vision.py` | body | Optional: local SmolVLM via llama.cpp for scene description. |
| Brain | `brain/nox_voice_brain.py` | brain | LLM loop: perception in → JSON with actions/speech out. OpenAI-compatible endpoint (incl. Ollama). |
| Body client | `brain/nox_body_client.py` | brain | Python API (`BodyClient` class + module functions) and CLI for the bridge. |
| Telegram bot | `brain/telegram_bot.py` | brain | Optional remote control channel. |

One systemd unit per component; `scripts/install-body.sh` /
`install-brain.sh` generate them for the local user and paths.

## Design principles

1. **Body = reflexes, brain = thought.** The robot handles fast local behavior
   (touch → wag, idle poses); reasoning and conversation live in the brain.
2. **Graceful degradation.** No brain reachable → the behavior engine keeps
   the dog autonomous. No voice model → voice stays off, everything else runs.
   No vision build → same. Optional means optional.
3. **Fail loudly.** Broken JSON, unknown actions, dead SDK threads, failing
   battery reads — every failure returns a diagnosable error instead of a
   silent `ok` (learned the hard way in issues #5–#12).
4. **HTTP is the contract.** Any hardware that implements `/action`, `/speak`
   and `/status` is a valid body (see `body/adapters/`); any client that
   speaks the API is a valid brain.
5. **LAN only.** No ports exposed to the internet; remote access via
   Tailscale (see [remote-access.md](remote-access.md)).

## Message flow

**Brain → body** (REST, see README API reference):

```json
POST /combo  {"actions": ["stand", "wag_tail"], "speak": "Hallo!",
              "rgb": {"r": 0, "g": 255, "b": 0, "mode": "breath"}}
```

**Body → brain** (perception push + voice input):

```json
{"type": "perception", "faces": [...], "objects": [...],
 "sensors": {"battery_v": 7.9, "touch": false}, "audio": {"speech": "..."}}
```

The brain answers voice input with a JSON action plan; if the LLM returns
malformed JSON, the dog speaks the raw reply and performs no actions —
graceful fallback, not a crash.

## Diagnostics

- `scripts/doctor.sh` — one-shot health check (role auto-detected), catches
  the known first-install traps.
- `GET /selftest` — drives the servos directly past the SDK's queue/thread
  machinery and reports process identity, thread health, and queue depth;
  localizes "commands succeed but nothing moves" failures to a layer.
- The daemon logs every external command — `journalctl -u nox-body` shows
  what actually arrived.
