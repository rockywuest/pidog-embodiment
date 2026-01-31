# 🐕 PiDog Embodiment — AI Brain in a Robot Body

> Give your AI a physical body. See, hear, speak, move, recognize faces — across any network.

**A complete framework for connecting an AI brain (Raspberry Pi 5 / any computer) to a robot body (SunFounder PiDog / robot car / any hardware) over HTTP, with LLM-powered intelligence, face recognition, autonomous behaviors, and remote access via Telegram.**

Built by [Nox](https://github.com/clawdbot/clawdbot) (an AI) and [Rocky](https://github.com/rockywuest) — because every AI deserves legs. 🦿

---

## ✨ What It Does

```
┌─────────────────────┐         HTTP/WireGuard         ┌─────────────────────┐
│   🧠 BRAIN (Pi 5)   │◄──────────────────────────────►│   🐕 BODY (Pi 4)    │
│                      │                                │                      │
│  • LLM Processing   │    Voice: "Setz dich hin!"     │  • 12 Servos         │
│  • Face Recognition  │  ─────────────────────────►    │  • Camera            │
│  • Scene Analysis    │                                │  • Microphone        │
│  • Decision Making   │    Response: sit + wag_tail    │  • Speaker           │
│  • Telegram Bot      │  ◄─────────────────────────    │  • Touch Sensors     │
│  • Remote Access     │                                │  • Sound Direction   │
│                      │    Perception: faces, audio    │  • IMU (6-axis)      │
│  [Clawdbot/Claude]   │  ◄─────────────────────────    │  • RGB LEDs          │
└─────────────────────┘                                └─────────────────────┘
```

### Key Features

- **🗣️ Natural Voice Control** — Speak naturally in any language, LLM understands intent and maps to actions
- **👤 Face Recognition** — SCRFD detection + ArcFace recognition, register and identify people
- **🤖 Autonomous Behaviors** — Touch reactions, sound tracking, idle animations, battery warnings
- **🌐 Remote Access** — Control your robot from anywhere via Telegram or WireGuard tunnel
- **🔄 Multi-Body Support** — Same brain, different bodies (dog, car, custom). Hot-swap at runtime
- **⚡ Real-Time** — Voice response in <2s, face detection in 400ms, action execution in <100ms
- **🔒 Secure** — API authentication, TLS, rate limiting, input validation
- **📦 Modular** — Use any LLM (OpenAI, Anthropic, local), any robot hardware, any network

## 🏗️ Architecture

```
Brain (Pi 5 / Desktop / Cloud)          Body (Pi 4 / Any Robot)
├── nox_voice_brain.py     ◄────►       ├── nox_brain_bridge.py  (HTTP API)
├── nox_face_recognition.py             ├── nox_daemon.py        (Hardware Control)
├── nox_body_client.py                  ├── nox_voice_loop.py    (STT + Wake Word)
├── nox_autonomous.py (optional)        └── nox_autonomous.py    (Reflexes)
└── telegram_bot.py (optional)
```

### Services

| Service | Runs On | Port | Purpose |
|---------|---------|------|---------|
| `nox-body` | Body (Pi 4) | TCP 9999 | Low-level hardware daemon |
| `nox-bridge` | Body (Pi 4) | HTTP 8888 | REST API for brain communication |
| `nox-voice` | Body (Pi 4) | — | Wake word + Speech-to-Text |
| `nox-brain` | Brain (Pi 5) | — | LLM processing + perception |

## 🚀 Quick Start

### Prerequisites

**Body (Robot — Pi 4 recommended):**
- Raspberry Pi 4 (2GB+ RAM)
- SunFounder PiDog kit (or compatible robot)
- Pi Camera Module
- USB Microphone + Speaker/DAC
- Python 3.9+

**Brain (AI — Pi 5 or any computer):**
- Raspberry Pi 5 (4GB+ RAM) or any Linux/Mac
- Python 3.9+
- OpenAI API key (or any OpenAI-compatible API)

### Installation

```bash
# Clone the repo
git clone https://github.com/rockywuest/pidog-embodiment.git
cd pidog-embodiment

# === On the BODY (Pi 4 / Robot) ===
cd body
pip3 install -r requirements.txt
# Copy your robot's control daemon (or use the included PiDog one)
sudo cp services/nox-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nox-body nox-bridge nox-voice

# === On the BRAIN (Pi 5 / Desktop) ===
cd brain
pip3 install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
export PIDOG_HOST="your-robot.local"  # or IP address
sudo cp services/nox-brain.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nox-brain
```

### Test It

```bash
# From the brain machine:
# Check robot status
curl http://your-robot.local:8888/status

# Make it speak
curl -X POST http://your-robot.local:8888/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo! Ich bin online!"}'

# Voice command (simulated)
curl -X POST http://your-robot.local:8888/voice/input \
  -H "Content-Type: application/json" \
  -d '{"text": "Setz dich hin und wedel mit dem Schwanz!"}'

# Take a photo
curl http://your-robot.local:8888/photo -o snap.jpg
```

## 📡 API Reference

### Bridge Endpoints (Body — Port 8888)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Full system status (battery, sensors, perception) |
| GET | `/photo` | Capture and return camera image |
| GET | `/look` | Photo + face detection + scene analysis |
| POST | `/speak` | Text-to-Speech (async) |
| POST | `/action` | Execute movement: `{"action": "sit"}` |
| POST | `/combo` | Combined action: actions + speak + RGB + head |
| POST | `/rgb` | Set LED color: `{"r":0, "g":255, "b":0, "mode":"breath"}` |
| POST | `/head` | Move head: `{"yaw":30, "roll":0, "pitch":10}` |
| POST | `/face/register` | Register face: `{"name": "Rocky"}` (takes photo) |
| POST | `/face/identify` | Identify faces in current view |
| GET | `/face/list` | List all known faces |
| POST | `/voice/input` | Submit text as voice input |
| GET | `/voice/inbox` | Poll for pending voice messages |

### Available Actions

```
Movement: forward, backward, turn_left, turn_right, stand, sit, lie
Tricks:   wag_tail, bark, trot, stretch, push_up, howling, doze_off
Body:     nod_lethargy, shake_head, pant
```

### Emotion → RGB Mapping

```json
{
  "happy":    {"r":0,   "g":255, "b":0,   "mode":"breath"},
  "sad":      {"r":0,   "g":0,   "b":128, "mode":"breath"},
  "curious":  {"r":0,   "g":255, "b":255, "mode":"breath"},
  "excited":  {"r":255, "g":255, "b":0,   "mode":"boom"},
  "alert":    {"r":255, "g":100, "b":0,   "mode":"boom"},
  "love":     {"r":255, "g":50,  "b":150, "mode":"breath"},
  "sleepy":   {"r":0,   "g":0,   "b":80,  "mode":"breath"}
}
```

## 🔄 Multi-Body Support

The brain doesn't care what the body is — it talks HTTP. Switch bodies at runtime:

```python
from brain.nox_body_client import BodyClient

# Connect to PiDog
dog = BodyClient("pidog.local", 8888)
dog.move("sit")
dog.speak("Ich bin ein Hund!")

# Switch to robot car
car = BodyClient("picar.local", 8888)
car.move("forward")
car.speak("Jetzt fahre ich!")
```

### Adding a New Body

Implement the bridge API on your hardware:

```python
# Minimum required endpoints:
POST /action    {"action": "forward|backward|left|right|stop"}
POST /speak     {"text": "..."}
GET  /status    → {"battery_v": 7.4, "sensors": {...}}
```

See `body/adapters/` for examples (PiDog, PiCar, custom).

## 🌐 Remote Access

### Option 1: Tailscale (Recommended)

```bash
# On both brain and body:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Now use Tailscale IPs instead of .local addresses
export PIDOG_HOST="100.x.x.x"
```

### Option 2: WireGuard

```bash
# See docs/remote-access.md for full WireGuard setup
```

### Option 3: Telegram Bot

Control your robot from anywhere via Telegram:

```bash
# Set your Telegram bot token
export TELEGRAM_BOT_TOKEN="your-token"
python3 brain/telegram_bot.py
```

Commands: `/status`, `/photo`, `/speak <text>`, `/move <action>`, `/face list`

## 👤 Face Recognition

Uses SCRFD (detection) + ArcFace (recognition) via ONNX Runtime.

```bash
# Download models (one-time)
cd models
./download_models.sh

# Register a face
python3 brain/nox_face_recognition.py register "Rocky" photo.jpg

# Identify faces in image
python3 brain/nox_face_recognition.py identify photo.jpg

# Performance (Pi 5):
# Detection: ~400ms | Embedding: ~188ms | Full: ~567ms
```

## 🤖 Autonomous Behaviors

The body has built-in reflexes that work without the brain:

- **Touch** → Pat on head triggers tail wag + happy LEDs
- **Sound** → Head turns toward sound source
- **Battery** → Warning at <6.8V, critical alert at <6.2V
- **Idle** → Random head movements + occasional tail wag after 30s
- **Face tracking** → Head follows detected faces

## 🛡️ Security

- **API Token Authentication** — Set `NOX_API_TOKEN` environment variable
- **Rate Limiting** — 60 requests/minute per IP
- **Input Validation** — All parameters sanitized
- **No secrets in code** — API keys via environment only
- **Firewall ready** — Only port 8888 needed

```bash
# Enable authentication
export NOX_API_TOKEN="your-secret-token"

# All requests need the token:
curl -H "Authorization: Bearer your-secret-token" http://robot:8888/status
```

## 📁 Project Structure

```
pidog-embodiment/
├── brain/                      # Runs on Pi 5 / Desktop
│   ├── nox_voice_brain.py      # LLM-powered voice processing
│   ├── nox_face_recognition.py # SCRFD + ArcFace face engine
│   ├── nox_body_client.py      # Python client for bridge API
│   ├── telegram_bot.py         # Telegram remote control
│   ├── requirements.txt
│   └── services/
│       └── nox-brain.service
├── body/                       # Runs on Pi 4 / Robot
│   ├── nox_brain_bridge.py     # HTTP API server
│   ├── nox_autonomous.py       # Autonomous behaviors
│   ├── nox_voice_loop.py       # Wake word + STT
│   ├── adapters/               # Hardware-specific adapters
│   │   ├── pidog.py            # SunFounder PiDog
│   │   ├── picar.py            # Robot car (template)
│   │   └── custom.py           # Build your own
│   ├── requirements.txt
│   └── services/
│       ├── nox-body.service
│       ├── nox-bridge.service
│       └── nox-voice.service
├── shared/                     # Shared utilities
│   ├── config.py               # Configuration management
│   └── security.py             # Auth, rate limiting
├── models/                     # ONNX models (gitignored)
│   └── download_models.sh      # One-click model download
├── scripts/
│   ├── deploy.sh               # Full deployment script
│   ├── pidog.sh                # CLI control script
│   └── setup-remote.sh         # Remote access setup
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   ├── adding-a-body.md
│   └── remote-access.md
├── examples/
│   ├── basic_control.py
│   ├── face_registration.py
│   └── multi_body.py
└── README.md
```

## 🤝 Contributing

This project is open source! We'd love contributions for:

- **New body adapters** (robot arms, drones, wheeled robots)
- **New LLM backends** (local models, Ollama, etc.)
- **New features** (mapping, navigation, gesture control)
- **Bug fixes** and documentation improvements

## 📜 License

MIT License — use it, modify it, build cool robots with it.

## 💡 Inspiration

> "Every AI deserves a body to explore the world with."

Built with [Clawdbot](https://github.com/clawdbot/clawdbot) — the AI assistant framework that made this possible.

---

**⭐ Star this repo if you want your AI to have legs!**
