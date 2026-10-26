"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_quiet_hours(monkeypatch):
    """The behavior engine's quiet hours (22-7) follow the wall clock — without
    this, engine tests fail whenever the suite runs at night. Tests about
    quiet hours patch them explicitly."""
    try:
        import body.nox_behavior_engine as be
    except Exception:
        return
    monkeypatch.setattr(be, "QUIET_HOURS", None)
