"""'Servos off' must cut the PWM signal (field: a sleeping dog hummed in the
hand with every Nox service stopped). Source-level: the daemon needs the SDK."""
import re

SRC = open("body/nox_daemon.py").read()


def _body(name):
    m = re.search(rf"\ndef {name}\(.*?(?=\ndef |\nif __name__)", SRC, re.S)
    assert m, name
    return m.group(0)


def test_release_writes_pulse_width_zero_on_every_part():
    body = _body("_release_servos")
    assert "pulse_width(0)" in body
    assert '("legs", "head", "tail")' in body
    assert "_action_buffer_depth()" in body  # a queued "lie" finishes first


def test_deep_sleep_sleep_emergency_and_shutdown_release():
    assert "_release_servos()" in _body("_idle_watchdog")
    assert "_release_servos()" in _body("cmd_sleep")
    assert "_release_servos()" in _body("cmd_emergency_stop")
    assert "_release_servos()" in _body("shutdown")


def test_release_comes_after_lying_down():
    for name in ("cmd_sleep", "cmd_emergency_stop", "shutdown"):
        body = _body(name)
        assert body.index("lie") < body.index("_release_servos()"), name
