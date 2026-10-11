# Security Policy

**Please do not open a public issue for a vulnerability.** Use GitHub's private
reporting: *Security → Report a vulnerability* on this repository. You'll get a
response within a few days, and credit in the fix unless you prefer otherwise.

## What this project does for security

- **No secrets in the tree** — enforced by `tests/test_no_private_data.py`
  (tokens, private addresses, personal data), which runs in CI on every change.
- **The bridge moves a physical robot.** Keep port 8888 on LAN/VPN only, and set
  `NOX_API_TOKEN` (see *Security* in the README): with CORS `*`, the token is
  the only barrier between any browser on your network and the robot.
- **Offline by design** — no telemetry, no phone-home; cloud LLMs are opt-in.
- **Input validation** on every endpoint that reaches servos, LEDs or the
  speaker; `/command` is a documented raw passthrough.

Current release: see [Releases](https://github.com/rockywuest/pidog-embodiment/releases).
Only the latest release receives fixes.
