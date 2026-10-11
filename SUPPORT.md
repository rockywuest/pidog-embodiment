# Getting Help

**First stop, always:** `./scripts/doctor.sh` on the affected machine. It checks
the SDK, I2C, services, audio, microphone, voice models and vision, and prints a
fix hint for everything that's off.

Opening an issue? Paste the full doctor output plus
`journalctl -u nox-body -n 50 --no-pager` (or `nox-brain` / `nox-voice` /
`nox-vision`) — that usually turns a week of back-and-forth into one reply.

## The four most common traps

1. **The dog moves but makes no sound** → the speaker is the robot HAT's own
   card, not card 0. `aplay -l` should list a `voicehat`/`googlevoi` card;
   `AUDIODEV=auto` in `body/nox.env` makes the daemon pick it and log why.
2. **Voice input stays off** → it needs a Vosk model for YOUR language and a
   microphone. `VOSK_MODEL_PATH` in `body/nox.env` points at the unzipped model
   folder (small models only — a 1 GB model does not fit on a Pi 4). The robot
   HAT's own mics are detected automatically.
3. **`/vision` says "not running"** → the vision service is opt-in:
   `sudo ./scripts/install-body.sh nox-vision`. The first result takes about a
   minute (the camera cold-starts).
4. **The dog answers in the wrong language or chats nonsense** → fix the reply
   language with `NOX_LANG=fr` (or de/en) on the brain, or switch to
   deterministic keyword commands with `NOX_MODE=commands`. The brain logs both
   at startup: `journalctl -u nox-brain | grep Language`.

## No hardware?

`./install.sh --mock` runs a pretend dog on `127.0.0.1:8888` — every endpoint,
no robot. Point the brain or the MCP server at it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests run anywhere:
`PYTHONPATH=. python3 -m pytest tests/ -q`.
