#!/usr/bin/env python3
"""
config.py — Centralized configuration for the embodiment system.

All settings can be overridden via environment variables.
"""

import os


def env(key, default=None, cast=None):
    """Get environment variable with optional type casting."""
    val = os.environ.get(key, default)
    if val is None:
        return val
    if cast:
        return cast(val)
    return val


class BrainConfig:
    """Configuration for the brain side."""
    
    # Connection to body
    BODY_HOST = env("PIDOG_HOST", "pidog.local")
    BODY_PORT = env("PIDOG_BRIDGE_PORT", 8888, int)
    BODY_URL = f"http://{BODY_HOST}:{BODY_PORT}"
    
    # LLM API (OpenAI-compatible)
    LLM_API_KEY = env("OPENAI_API_KEY", "")
    LLM_API_URL = env("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")
    LLM_MAX_TOKENS = env("LLM_MAX_TOKENS", 256, int)
    LLM_TEMPERATURE = env("LLM_TEMPERATURE", 0.7, float)
    
    # Voice processing
    POLL_INTERVAL = env("POLL_INTERVAL", 0.8, float)
    SENSOR_CHECK_INTERVAL = env("SENSOR_CHECK_INTERVAL", 15.0, float)
    CONVERSATION_HISTORY = env("CONVERSATION_HISTORY", 8, int)
    
    # Face recognition
    MODEL_DIR = env("MODEL_DIR", os.path.expanduser("~/models"))
    FACE_DB_DIR = env("FACE_DB_DIR", os.path.expanduser("~/face_db"))
    FACE_DETECT_THRESHOLD = env("FACE_DETECT_THRESHOLD", 0.5, float)
    FACE_RECOGNIZE_THRESHOLD = env("FACE_RECOGNIZE_THRESHOLD", 0.4, float)
    
    # Telegram bot
    TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_ALLOWED_USERS = env("TELEGRAM_ALLOWED_USERS", "")  # comma-separated IDs


class BodyConfig:
    """Configuration for the body side."""
    
    # Bridge server
    LISTEN_HOST = env("BRIDGE_HOST", "0.0.0.0")
    LISTEN_PORT = env("BRIDGE_PORT", 8888, int)
    
    # Security
    API_TOKEN = env("NOX_API_TOKEN", "")
    RATE_LIMIT = env("RATE_LIMIT", 60, int)
    RATE_WINDOW = env("RATE_WINDOW", 60, int)
    
    # Hardware daemon
    DAEMON_HOST = env("DAEMON_HOST", "localhost")
    DAEMON_PORT = env("DAEMON_PORT", 9999, int)
    
    # Camera
    CAMERA_WIDTH = env("CAMERA_WIDTH", 640, int)
    CAMERA_HEIGHT = env("CAMERA_HEIGHT", 480, int)
    
    # Voice
    WAKE_WORD = env("WAKE_WORD", "nox")
    STT_MODEL = env("STT_MODEL", "vosk-model-de-0.21")  # or "whisper-small"
    TTS_VOICE = env("TTS_VOICE", "thorsten")
    
    # Autonomous behaviors
    IDLE_TIMEOUT = env("IDLE_TIMEOUT", 30.0, float)
    SENSOR_INTERVAL = env("SENSOR_INTERVAL", 0.5, float)
    BATTERY_LOW = env("BATTERY_LOW", 6.8, float)
    BATTERY_CRITICAL = env("BATTERY_CRITICAL", 6.2, float)


class RemoteConfig:
    """Configuration for remote access."""
    
    # Tailscale
    TAILSCALE_ENABLED = env("TAILSCALE_ENABLED", "false") == "true"
    
    # WireGuard
    WIREGUARD_ENABLED = env("WIREGUARD_ENABLED", "false") == "true"
    WIREGUARD_CONFIG = env("WIREGUARD_CONFIG", "/etc/wireguard/wg0.conf")
    
    # Telegram
    TELEGRAM_ENABLED = env("TELEGRAM_ENABLED", "false") == "true"
