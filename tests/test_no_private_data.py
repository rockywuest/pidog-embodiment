"""Guard against private data in a public repository.

A 48-hex gateway token sat in brain/nox_voice_relay.py as a default value, and
went public with the commit that added it. Alongside it: the owner's tailnet
addresses, their LAN subnet, their router's IP, and the first names and ages of
three children — all as defaults in executable code.

Working defaults are convenient for exactly one person and a liability for
everyone else, so this test fails on the patterns that caused it. If it fires on
something legitimate, add it to ALLOWED with a reason rather than loosening a
pattern.
"""
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Files whose job is to show example values.
EXAMPLE_FILES = {"body/nox.env.example", "README.md", "CONTRIBUTING.md"}
ALLOWED = {
    # doctor.sh explains the LAN/Tailscale firewall ranges it checks for
    ("docs/remote-access.md", "100.64.0.0"): "documented Tailscale CGNAT range",
    ("docs/remote-access.md", "100.100.100.100"): "Tailscale's own DNS address",
}

SECRET_PATTERNS = (
    # 32+ hex chars in a row: tokens, keys, session ids
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])"), "long hex string (token/key?)"),
    (re.compile(r"(?i)\b(sk|pk|ghp|gho|xox[bap])[-_][A-Za-z0-9]{16,}"), "provider API key"),
    (re.compile(r"(?i)(token|secret|password|api_key)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}[\"']"), "inline credential"),
)

# Tailscale/CGNAT 100.64.0.0/10 and any hardcoded private LAN address.
NETWORK_PATTERNS = (
    (re.compile(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"), "tailnet address"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "private LAN address"),
    (re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "private LAN address"),
)

# Names of the owner's household, which used to ship inside the system prompts.
PERSONAL_PATTERN = re.compile(
    r"(?i)\b(Bea|Noah|Klara|Eliah)\b\s*\(?\d{0,2}\)?\s*(,|\)|$|\s)")


SELF = "tests/" + Path(__file__).name


def tracked_files(*suffixes):
    """Tracked files with these extensions, minus this file — it quotes the
    patterns it searches for and would flag itself."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n")
            if f and f.endswith(suffixes) and f != SELF]


COMMENT_STARTS = ("#", '"""', "'''", "*", "//")


def offending_lines(path, patterns, code_only=False):
    hits = []
    text = (REPO / path).read_text(errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        if "test_no_private_data" in line:
            continue
        # A documented example inside a comment or usage text is fine. What must
        # never exist is a real address baked into executable code.
        if code_only and line.lstrip().startswith(COMMENT_STARTS):
            continue
        for pattern, label in patterns:
            for match in pattern.findall(line):
                value = match if isinstance(match, str) else match[0]
                if (path, value) in ALLOWED:
                    continue
                if any(k[0] == path and value.startswith(k[1]) for k in ALLOWED):
                    continue
                hits.append(f"{path}:{lineno}  {label}: {line.strip()[:110]}")
    return hits


@pytest.mark.parametrize("path", tracked_files(".py", ".sh", ".service", ".json", ".yml"))
def test_no_secrets_in_code(path):
    hits = offending_lines(path, SECRET_PATTERNS)
    assert not hits, "secret-looking value in a public repo:\n" + "\n".join(hits)


@pytest.mark.parametrize("path", tracked_files(".py", ".sh", ".service"))
def test_no_hardcoded_private_addresses(path):
    """Executable code must not carry anyone's real network as a default."""
    if path in EXAMPLE_FILES:
        pytest.skip("example file")
    hits = offending_lines(path, NETWORK_PATTERNS, code_only=True)
    assert not hits, ("hardcoded private address — use a hostname default or an "
                      "env var:\n" + "\n".join(hits))


@pytest.mark.parametrize("path", tracked_files(".py", ".md", ".sh", ".service"))
def test_no_household_names(path):
    """Real names, children's above all, come from NOX_HOUSEHOLD at runtime."""
    text = (REPO / path).read_text(errors="replace")
    hits = [line.strip()[:110] for line in text.splitlines()
            if PERSONAL_PATTERN.search(line) and "NOX_HOUSEHOLD" not in line]
    assert not hits, "household names in the repo:\n" + "\n".join(hits)


# The gateway token that was published in e6ae049, stored as a digest: a guard
# that quotes the secret it guards against puts it back into the current tree,
# where secret scanners and anyone reading the tests would find it.
LEAKED_TOKEN_SHA256 = "9b1cae767938ae38507c67febbbdaf708ba894305466dbf64355d2132781d745"
HEX_RUN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def test_the_leaked_token_is_gone():
    """A revert or a stale branch must not bring the published token back."""
    for path in tracked_files(".py", ".sh", ".service", ".md", ".json", ".txt", ".yml"):
        for candidate in HEX_RUN.findall((REPO / path).read_text(errors="replace")):
            digest = hashlib.sha256(candidate.lower().encode()).hexdigest()
            assert digest != LEAKED_TOKEN_SHA256, (
                f"{path}: the gateway token published in e6ae049 is back")


def test_the_digest_mechanism_detects_a_token(monkeypatch):
    """The mechanism is verified with a made-up token, so the real one stays out
    of this file: it is extracted from a source line and matched by digest."""
    fake = "0123456789abcdef" * 3
    line = f'GATEWAY_TOKEN = os.environ.get("CLAWDBOT_TOKEN", "{fake}")'
    assert HEX_RUN.findall(line) == [fake], "a token default must be extractable"
    digest = hashlib.sha256(fake.encode()).hexdigest()
    assert digest != LEAKED_TOKEN_SHA256
    # and a digest comparison flags it when it is the watched value
    monkeypatch.setattr("tests.test_no_private_data.LEAKED_TOKEN_SHA256", digest)
    assert hashlib.sha256(fake.encode()).hexdigest() == digest
