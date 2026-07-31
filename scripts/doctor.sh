#!/usr/bin/env bash
# doctor.sh — one-shot health check for a PiDog Embodiment install.
#
# Every check maps to a real first-install trap (issues #5, #10, #12): wrong
# home paths, missing cmake, absent voice models, battery not powering the
# servos, placeholder nox.env. Run it on the BODY (the robot) and/or the BRAIN;
# it auto-detects which side it's on and only runs the relevant checks.
#
# Usage:  ./scripts/doctor.sh          # no sudo needed
#
# Exit code: 0 if nothing FAILED (warnings are fine), 1 if any check FAILED.

# Deliberately no `set -e`: we want every check to run even when earlier ones fail.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BODY_DIR="$REPO_DIR/body"
BRIDGE_PORT="${PIDOG_BRIDGE_PORT:-8888}"

if [[ -t 1 ]]; then
  G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=""; Y=""; R=""; B=""; N=""
fi

fails=0; warns=0
pass() { printf '  %sPASS%s  %s\n' "$G" "$N" "$1"; }
warn() { printf '  %sWARN%s  %s\n' "$Y" "$N" "$1"; warns=$((warns+1)); }
fail() { printf '  %sFAIL%s  %s\n' "$R" "$N" "$1"; fails=$((fails+1)); }
hint() { printf '        %s↳ %s%s\n' "$B" "$1" "$N"; }
section() { printf '\n%s%s%s\n' "$B" "$1" "$N"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ── Role detection ────────────────────────────────────────────────────────
# Body = SunFounder SDK importable OR body units installed.
# Brain = nox-brain unit installed OR /etc/default/nox-brain present.
IS_BODY=0; IS_BRAIN=0
if python3 -c "import pidog" >/dev/null 2>&1 || [[ -f /etc/systemd/system/nox-body.service ]]; then
  IS_BODY=1
fi
if [[ -f /etc/systemd/system/nox-brain.service || -f /etc/default/nox-brain ]]; then
  IS_BRAIN=1
fi
if [[ $IS_BODY -eq 0 && $IS_BRAIN -eq 0 ]]; then
  # Nothing installed yet — check the body side by default (most common first run).
  IS_BODY=1
fi

printf '%s🐕 PiDog Embodiment — doctor%s   (repo: %s)\n' "$B" "$N" "$REPO_DIR"
printf 'Detected role:%s%s%s\n' "$B" \
  "$([[ $IS_BODY -eq 1 ]] && printf ' body') $([[ $IS_BRAIN -eq 1 ]] && printf ' brain')" "$N"

svc_active() { systemctl is-active --quiet "$1" 2>/dev/null; }

check_service() {
  local unit="$1"
  if [[ ! -f "/etc/systemd/system/${unit}.service" ]]; then
    warn "$unit is not installed"
    hint "run: sudo ./scripts/install-body.sh (body) or install-brain.sh (brain)"
  elif svc_active "$unit"; then
    pass "$unit is running"
  else
    fail "$unit is installed but not running"
    hint "sudo systemctl restart $unit && journalctl -u $unit -n 30 --no-pager"
  fi
}

# ── BODY checks ───────────────────────────────────────────────────────────
if [[ $IS_BODY -eq 1 ]]; then
  section "Body — hardware & SDK"

  if python3 -c "import pidog, robot_hat" >/dev/null 2>&1; then
    ver=$(python3 -c "import pidog; print(getattr(pidog,'__version__','?'))" 2>/dev/null)
    pass "SunFounder SDK importable (pidog ${ver:-?})"
  else
    fail "SunFounder pidog/robot_hat SDK not importable"
    hint "install it from https://github.com/sunfounder/pidog (not on PyPI alone)"
  fi

  section "Body — configuration"
  if [[ -f "$BODY_DIR/nox.env" ]]; then
    if grep -q '[<>]' "$BODY_DIR/nox.env"; then
      fail "body/nox.env still contains <PLACEHOLDER> values"
      hint "edit BRAIN_HOST (127.0.0.1 if brain+body share one machine)"
    elif grep -qE '^BRAIN_HOST=' "$BODY_DIR/nox.env"; then
      pass "body/nox.env present, BRAIN_HOST set ($(grep -E '^BRAIN_HOST=' "$BODY_DIR/nox.env" | head -1 | cut -d= -f2))"
    else
      warn "body/nox.env present but BRAIN_HOST not set"
    fi
  else
    warn "body/nox.env missing (install-body.sh creates it from the example)"
  fi

  section "Body — services"
  check_service nox-body
  check_service nox-bridge
  check_service nox-voice

  section "Body — bridge API"
  if have curl; then
    status_json="$(curl -s --max-time 5 "http://127.0.0.1:${BRIDGE_PORT}/status" 2>/dev/null)"
    if [[ -n "$status_json" ]]; then
      pass "bridge answers on :${BRIDGE_PORT}/status"
      if printf '%s' "$status_json" | grep -q '"battery_v": *"error"'; then
        berr="$(printf '%s' "$status_json" | sed -n 's/.*"battery_error": *"\([^"]*\)".*/\1/p')"
        warn "battery_v is \"error\" — the daemon can't READ the battery voltage${berr:+ (${berr})}"
        hint "if the dog also won't move: check the 2-cell battery — servos run on it, NOT USB-C"
        hint "if the dog moves fine, only the ADC read fails; test it directly:"
        hint "  python3 -c 'from robot_hat import utils; print(utils.get_battery_voltage())'"
        [[ -z "$berr" ]] && hint "no battery_error detail — update + restart: git pull && sudo systemctl restart nox-body"
      elif printf '%s' "$status_json" | grep -qE '"battery_v": *[0-9]'; then
        pass "battery voltage readable ($(printf '%s' "$status_json" | grep -oE '"battery_v": *[0-9.]+' | head -1 | grep -oE '[0-9.]+')V)"
      fi

      # Is the RUNNING bridge process on current code? An empty /action POST
      # must error loudly (issue #13); a silent ok means the service was never
      # restarted after git pull. Harmless probe: no action is ever executed.
      action_probe="$(curl -s --max-time 5 -X POST "http://127.0.0.1:${BRIDGE_PORT}/action" \
        -H 'Content-Type: application/json' -d '{}' 2>/dev/null)"
      if printf '%s' "$action_probe" | grep -q 'no action given'; then
        pass "running bridge is on current code (/action errors loudly on empty input)"
      elif printf '%s' "$action_probe" | grep -q '"ok": *true'; then
        fail "running bridge is on OUTDATED code — /action still swallows empty input silently"
        hint "cd $REPO_DIR && git pull && sudo systemctl restart nox-bridge nox-body"
      fi
    else
      fail "no response from bridge on :${BRIDGE_PORT}"
      hint "is nox-bridge running? sudo systemctl restart nox-bridge"
    fi
  else
    warn "curl not installed — skipping live API check"
  fi

  section "Body — voice output (Piper TTS)"
  piper_bin="${PIPER_BIN:-}"
  [[ -z "$piper_bin" ]] && { [[ -x "$HOME/.local/bin/piper" ]] && piper_bin="$HOME/.local/bin/piper"; }
  [[ -z "$piper_bin" ]] && piper_bin="$(command -v piper 2>/dev/null || true)"
  if [[ -n "$piper_bin" && -x "$piper_bin" ]]; then
    pass "piper binary found ($piper_bin)"
  else
    warn "piper binary not found — /speak will report an error"
    hint "pip3 install piper-tts (or set PIPER_BIN in body/nox.env)"
  fi
  piper_model="${PIPER_MODEL:-$HOME/.local/share/piper-voices/de_DE-thorsten-high.onnx}"
  if [[ -f "$piper_model" ]]; then
    pass "piper voice model present ($(basename "$piper_model"))"
  else
    warn "no piper voice model at $piper_model"
    hint "download one from https://huggingface.co/rhasspy/piper-voices"
    hint "then set PIPER_MODEL in body/nox.env"
  fi

  section "Body — voice input (Vosk STT, optional)"
  vosk_model="${VOSK_MODEL_PATH:-$HOME/vosk-models/vosk-model-small-de-0.15}"
  if [[ -d "$vosk_model" ]]; then
    pass "Vosk model present ($(basename "$vosk_model"))"
  else
    warn "no Vosk model — voice input stays off (nox-voice exits cleanly)"
    hint "optional: download from https://alphacephei.com/vosk/models, set VOSK_MODEL_PATH"
  fi

  section "Body — local vision (SmolVLM, optional)"
  if [[ -x "$HOME/llama.cpp/build/bin/llama-mtmd-cli" || -x "$HOME/llama.cpp/build/bin/llama-llava-cli" ]]; then
    pass "llama.cpp built"
  else
    warn "llama.cpp not built — vision disabled (everything else works)"
    have cmake || hint "prerequisite missing: sudo apt install -y cmake build-essential"
    hint "see the 'Local Vision' section in the README"
  fi
  if [[ -f "$HOME/models/smolvlm/SmolVLM-256M-Instruct-Q8_0.gguf" ]]; then
    pass "SmolVLM model present"
  else
    warn "SmolVLM model not downloaded — vision disabled"
  fi
fi

# ── BRAIN checks ──────────────────────────────────────────────────────────
if [[ $IS_BRAIN -eq 1 ]]; then
  section "Brain — configuration"
  if [[ -f /etc/default/nox-brain ]]; then
    pass "/etc/default/nox-brain present"
    if grep -qE '^[[:space:]]*PIDOG_HOST=' /etc/default/nox-brain; then
      pass "PIDOG_HOST set ($(grep -E '^[[:space:]]*PIDOG_HOST=' /etc/default/nox-brain | head -1 | cut -d= -f2))"
    else
      warn "PIDOG_HOST not set in /etc/default/nox-brain"
      hint "set it to the robot's IP so the brain can reach the body on :${BRIDGE_PORT}"
    fi
  else
    warn "/etc/default/nox-brain missing"
    hint "run: sudo ./scripts/install-brain.sh"
  fi

  section "Brain — service"
  check_service nox-brain

  section "Brain — reach the body"
  pidog_host="$(grep -E '^[[:space:]]*PIDOG_HOST=' /etc/default/nox-brain 2>/dev/null | head -1 | cut -d= -f2)"
  pidog_host="${pidog_host:-pidog.local}"
  if have curl; then
    if curl -s --max-time 5 "http://${pidog_host}:${BRIDGE_PORT}/status" >/dev/null 2>&1; then
      pass "body reachable at ${pidog_host}:${BRIDGE_PORT}"
    else
      warn "cannot reach body at ${pidog_host}:${BRIDGE_PORT}"
      hint "run this doctor on the robot; confirm PIDOG_HOST and that both are on one network"
    fi
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
section "Summary"
if [[ $fails -eq 0 && $warns -eq 0 ]]; then
  printf '  %sAll checks passed. Nox is healthy. 🐕%s\n' "$G" "$N"
elif [[ $fails -eq 0 ]]; then
  printf '  %s%d warning(s), no failures.%s Optional features may be off; core works.\n' "$Y" "$warns" "$N"
else
  printf '  %s%d failure(s)%s and %d warning(s). Fix the FAILs above first.\n' "$R" "$fails" "$N" "$warns"
fi

[[ $fails -eq 0 ]]
