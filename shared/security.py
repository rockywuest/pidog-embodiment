#!/usr/bin/env python3
"""
security.py — Authentication, rate limiting, and input validation.

Designed to protect the bridge API when exposed to the internet.
"""

import os
import time
import hashlib
import hmac
import json
import re
from collections import defaultdict
from threading import Lock


class TokenAuth:
    """Bearer token authentication."""
    
    def __init__(self, token=None):
        self.token = token or os.environ.get("NOX_API_TOKEN", "")
        self.enabled = bool(self.token)
    
    def check(self, headers):
        """Check Authorization header. Returns (ok, error_msg)."""
        if not self.enabled:
            return True, None
        
        auth = headers.get("Authorization", "")
        if not auth:
            return False, "Missing Authorization header"
        
        if auth.startswith("Bearer "):
            provided = auth[7:]
        else:
            provided = auth
        
        if hmac.compare_digest(provided, self.token):
            return True, None
        
        return False, "Invalid token"


class RateLimiter:
    """Simple sliding window rate limiter."""
    
    def __init__(self, max_requests=60, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def check(self, client_ip):
        """Check if request is allowed. Returns (ok, retry_after_seconds)."""
        now = time.time()
        
        with self.lock:
            # Clean old entries
            cutoff = now - self.window
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if t > cutoff
            ]
            
            if len(self.requests[client_ip]) >= self.max_requests:
                oldest = self.requests[client_ip][0]
                retry_after = int(oldest + self.window - now) + 1
                return False, retry_after
            
            self.requests[client_ip].append(now)
            return True, 0
    
    def cleanup(self):
        """Remove stale entries (call periodically)."""
        cutoff = time.time() - self.window
        with self.lock:
            stale = [ip for ip, times in self.requests.items() 
                     if not times or times[-1] < cutoff]
            for ip in stale:
                del self.requests[ip]


class InputValidator:
    """Validate and sanitize API inputs."""
    
    # Allowed action names (whitelist)
    ALLOWED_ACTIONS = {
        "forward", "backward", "turn_left", "turn_right",
        "stand", "sit", "lie", "wag_tail", "bark", "trot",
        "doze_off", "stretch", "push_up", "howling", "pant",
        "nod_lethargy", "shake_head", "nod",
    }
    
    # Limits
    MAX_TEXT_LENGTH = 500
    MAX_NAME_LENGTH = 50
    RGB_RANGE = (0, 255)
    YAW_RANGE = (-80, 80)
    PITCH_RANGE = (-30, 30)
    ROLL_RANGE = (-30, 30)
    
    @classmethod
    def validate_action(cls, action):
        """Validate action name."""
        if not isinstance(action, str):
            return None, "action must be a string"
        action = action.strip().lower()
        if action not in cls.ALLOWED_ACTIONS:
            return None, f"Unknown action: {action}. Allowed: {sorted(cls.ALLOWED_ACTIONS)}"
        return action, None
    
    @classmethod
    def validate_text(cls, text, max_len=None):
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            return None, "text must be a string"
        text = text.strip()
        max_len = max_len or cls.MAX_TEXT_LENGTH
        if len(text) > max_len:
            return None, f"Text too long ({len(text)} > {max_len})"
        if not text:
            return None, "Text cannot be empty"
        # Remove any control characters except newline
        text = re.sub(r'[\x00-\x09\x0b-\x1f\x7f]', '', text)
        return text, None
    
    @classmethod
    def validate_rgb(cls, r, g, b):
        """Validate RGB values."""
        for name, val in [("r", r), ("g", g), ("b", b)]:
            if not isinstance(val, (int, float)):
                return None, f"{name} must be a number"
            if not (cls.RGB_RANGE[0] <= val <= cls.RGB_RANGE[1]):
                return None, f"{name} must be {cls.RGB_RANGE[0]}-{cls.RGB_RANGE[1]}"
        return (int(r), int(g), int(b)), None
    
    @classmethod
    def validate_head(cls, yaw=0, roll=0, pitch=0):
        """Validate head angles."""
        for name, val, (lo, hi) in [
            ("yaw", yaw, cls.YAW_RANGE),
            ("roll", roll, cls.ROLL_RANGE),
            ("pitch", pitch, cls.PITCH_RANGE),
        ]:
            if not isinstance(val, (int, float)):
                return None, f"{name} must be a number"
            if not (lo <= val <= hi):
                return None, f"{name} must be {lo} to {hi}"
        return (float(yaw), float(roll), float(pitch)), None
    
    @classmethod
    def validate_name(cls, name):
        """Validate a person/face name."""
        if not isinstance(name, str):
            return None, "name must be a string"
        name = name.strip()
        if not name or len(name) > cls.MAX_NAME_LENGTH:
            return None, f"Name must be 1-{cls.MAX_NAME_LENGTH} characters"
        if not re.match(r'^[\w\s\-\.]+$', name, re.UNICODE):
            return None, "Name contains invalid characters"
        return name, None


class SecurityMiddleware:
    """Combined security middleware for the bridge HTTP server."""
    
    def __init__(self, token=None, rate_limit=60, rate_window=60):
        self.auth = TokenAuth(token)
        self.limiter = RateLimiter(rate_limit, rate_window)
        self.validator = InputValidator()
        
        # Whitelist local IPs from rate limiting
        self.local_ips = {"127.0.0.1", "::1", "localhost"}
    
    def check_request(self, client_ip, headers):
        """Run all security checks. Returns (ok, status_code, error_msg)."""
        # Auth check
        ok, err = self.auth.check(headers)
        if not ok:
            return False, 401, err
        
        # Rate limit (skip for local)
        if client_ip not in self.local_ips:
            ok, retry = self.limiter.check(client_ip)
            if not ok:
                return False, 429, f"Rate limited. Retry after {retry}s"
        
        return True, 200, None
    
    def status(self):
        return {
            "auth_enabled": self.auth.enabled,
            "rate_limit": self.limiter.max_requests,
            "rate_window_s": self.limiter.window,
        }
