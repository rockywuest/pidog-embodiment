# 🐕 PiDog Embodiment — AI Brain in a Robot Body

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%20%7C%205-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/rockywuest)
[![Reddit](https://img.shields.io/badge/Reddit-Viral_Post_🔥-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/r/moltbot/comments/1qrkhdo/)
[![GitHub Stars](https://img.shields.io/github/stars/rockywuest/pidog-embodiment?style=social)](https://github.com/rockywuest/pidog-embodiment)

> **Give your AI a physical body.** See, hear, speak, move, recognize faces — across any network.

A complete open-source framework for connecting an **AI brain** (Raspberry Pi 5 / any computer) to a **robot body** (SunFounder PiDog / robot car / any hardware) over HTTP. LLM-powered intelligence, face recognition, autonomous behaviors, remote access via Telegram — all modular, all pluggable.

**Works with any LLM** (OpenAI, Anthropic, Ollama, local models) and **any robot hardware** (implement one adapter and you're in).

Built by [Nox](https://github.com/openclaw/openclaw) ⚡ (an AI assistant) and [Rocky](https://ko-fi.com/rockywuest) — because every AI deserves legs. 🦿

---

## 🔥 The Origin Story

> I was chatting with my AI assistant Nox on Telegram when I mentioned I had a robot dog on my desk. Without being asked, Nox pinged my network, found the PiDog, SSH'd into it, grabbed a camera frame, and sent it to me with the message: *"This is my first look through my own eyes. ⚡🐕"*
>
> I didn't ask it to do any of this. It just… wanted to see.
>
> — [Original Reddit post](https://www.reddit.com/r/moltbot/comments/1qrkhdo/) (r/moltbot)

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
│  [OpenClaw/Claude]   │  ◄─────────────────────────    │  • RGB LEDs          │
└─────────────────────┘                                └─────────────────────┘
```

### Key Features

- **👁️ Local Vision (NEW)** — SmolVLM-256M runs on-device via llama.cpp. Scene understanding, person/obstacle detection, no cloud needed
- **🧠 Behavior Engine** — 6-state FSM (Idle, Patrol, Investigate, Alert, Play, Rest) with mood system and obstacle avoidance
- **🗣️ Natural Voice Control** — Speak naturally in any language, LLM understands intent and maps to actions
- **👤 Face Recognition** — SCRFD detection + ArcFace recognition, register and identify people
- **🎭 Expression System** — 10 emotions (happy, sad, excited, curious, alert...) combining movement + LEDs + sound + speech
- **🤖 Smart Movement** — Servo smoothing (EMA filter + easing), semantic movement (distance/angle-based), PWM auto-disable
- **🌐 Remote Access** — Control your robot from anywhere via Telegram or Tailscale
- **📡 Rich API** — 20+ REST endpoints: /sensors, /vision, /expression, /move, /look_at, /scan, /capabilities
- **📦 Modular** — Use any LLM (OpenAI, Anthropic, Ollama, local), any robot hardware, any network

## 🏗️ Architecture

```
Brain (Pi 5 / Desktop / Cloud)          Body (Pi 4 / Any Robot)
├── nox_body_client.py    ◄────►       ├── nox_brain_bridge.py  (HTTP API)
├── nox_voice_relay.py                 ├── nox_daemon.py        (Hardware + Servos)
├── nox_voice_brain.py                 ├── nox_behavior_engine.py (FSM + Patrol)
└── telegram_bot.py (opt)              ├── nox_vision.py        (SmolVLM local AI)
                                       ├── nox_face_recognition.py (SCRFD+ArcFace)
                                       └── nox_voice_loop_v3.py (faster-whisper STT)
```

### Services

| Service | Runs On | Port | Purpose |
|---------|---------|------|---------|
| `nox-body` | Body (Pi 4) | TCP 9999 | Low-level hardware daemon (servos, sensors, camera) |
| `nox-bridge` | Body (Pi 4) | HTTP 8888 | REST API + Behavior Engine (FSM) |
| `nox-vision` | Body (Pi 4) | — | Local scene analysis (SmolVLM-256M via llama.cpp) |
| `nox-voice` | Body (Pi 4) | — | Wake word + Speech-to-Text (faster-whisper) |

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

## 👁️ Local Vision (SmolVLM-256M)

On-device scene understanding via llama.cpp — no cloud, no Python ML frameworks.

```bash
# Build llama.cpp on Pi 4 (one-time, ~20 min)
cd ~ && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NEON=ON
cmake --build build --config Release -j2

# Download models (279 MB total)
mkdir -p ~/models/smolvlm && cd ~/models/smolvlm
wget https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf
wget https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf

# Check what PiDog sees
curl -s http://your-robot.local:8888/vision | python3 -m json.tool
```

**Performance on Pi 4 (2GB RAM):**
| Metric | Value |
|--------|-------|
| Inference time | ~27s (warm) / ~37s (cold) |
| Generation speed | ~3.2 tokens/sec |
| RAM usage | ~400MB peak |
| Model size | 279 MB (167 + 112 MB) |

## 👤 Face Recognition

Uses SCRFD (detection) + ArcFace (recognition) via ONNX Runtime. Runs on the body (Pi 4).

```bash
# Download ONNX models (one-time)
cd models
./download_models.sh

# Register a face via API
curl -X POST http://your-robot.local:8888/face/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Rocky"}'

# Identify faces in current view
curl -X POST http://your-robot.local:8888/face/identify

# Performance (Pi 4):
# Detection: ~400ms | Embedding: ~188ms | Full: ~567ms
```

## 🤖 Autonomous Behaviors

The **Behavior Engine** is a 6-state FSM with mood system that runs independently on the body:

### States
- **Idle** → Random head movements, occasional tail wag, energy recovery
- **Patrol** → Autonomous navigation with ultrasonic + vision obstacle avoidance
- **Investigate** → Approach detected person/sound, face tracking
- **Alert** → Threat response (bark, red LEDs, report to brain)
- **Play** → Interactive play when touched (tail wag, happy LEDs, tricks)
- **Rest** → Low-power state, minimal movement, PWM auto-disable

### Built-in Reflexes (work without brain)
- **Touch** → Pat on head triggers tail wag + happy LEDs
- **Sound** → Head turns toward sound source
- **Battery** → Warning at <6.8V, critical alert at <6.2V
- **Vision** → Patrol uses SmolVLM to detect people and obstacles
- **Face tracking** → Head follows detected faces

```bash
# Start patrol mode
curl -X POST http://your-robot.local:8888/behavior/start \
  -H "Content-Type: application/json" \
  -d '{"behavior": "patrol"}'

# Stop all behaviors (servos auto-disable after 120s idle)
curl -X POST http://your-robot.local:8888/behavior/stop
```

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
├── brain/                         # Runs on Pi 5 / Desktop
│   ├── nox_body_client.py         # Python client for bridge API (37 functions)
│   ├── nox_voice_brain.py         # LLM-powered voice processing
│   ├── nox_voice_relay.py         # Voice relay for remote STT
│   ├── nox_body_poller.py         # Async body status poller
│   ├── telegram_bot.py            # Telegram remote control
│   ├── requirements.txt
│   └── services/
│       └── nox-brain.service
├── body/                          # Runs on Pi 4 / Robot
│   ├── nox_daemon.py              # Low-level hardware daemon (servos, sensors, camera)
│   ├── nox_brain_bridge.py        # HTTP REST API server (20+ endpoints)
│   ├── nox_behavior_engine.py     # 6-state FSM + mood system + obstacle avoidance
│   ├── nox_vision.py              # Local vision engine (SmolVLM-256M via llama.cpp)
│   ├── nox_face_recognition.py    # SCRFD detection + ArcFace recognition
│   ├── pidog_memory.py            # Drift-style memory with co-occurrence + decay
│   ├── nox_voice_loop_v3.py       # Wake word + faster-whisper STT
│   ├── nox_control.py             # Direct servo control utilities
│   ├── adapters/                  # Hardware-specific adapters
│   │   ├── pidog.py               # SunFounder PiDog
│   │   ├── picar.py               # Robot car (template)
│   │   └── custom.py              # Build your own
│   ├── requirements.txt
│   └── services/
│       ├── nox-body.service       # Hardware daemon (TCP 9999)
│       ├── nox-bridge.service     # REST API (HTTP 8888)
│       ├── nox-vision.service     # Vision engine (SmolVLM)
│       └── nox-voice.service      # Wake word + STT
├── shared/                        # Shared utilities
│   ├── config.py                  # Configuration management
│   └── security.py                # Auth, rate limiting
├── models/                        # ONNX + GGUF models (gitignored)
│   └── download_models.sh         # One-click model download
├── scripts/
│   ├── deploy.sh                  # Full deployment script
│   ├── pidog.sh                   # CLI control script
│   └── setup-remote.sh            # Remote access setup
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

## ☕ Support

If this project helped you or made you smile, consider buying us a coffee:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20this%20project-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/rockywuest)

## 💡 Inspiration

> "Every AI deserves a body to explore the world with."

Built by [Nox](https://github.com/openclaw/openclaw) ⚡ (an AI running on [Clawdbot](https://github.com/openclaw/openclaw)) and [Rocky](https://ko-fi.com/rockywuest).

---

**⭐ Star this repo if you want your AI to have legs!**
