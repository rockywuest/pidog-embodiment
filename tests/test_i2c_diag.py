"""I2C reachability diagnostics — issue #12.

The dog answered ok:true to every command, reported battery_v 0.0 and never
moved, while SunFounder's examples under sudo worked. robot_hat locates the
MCU with `i2cdetect` (in /usr/sbin, which sudo has on PATH and the service
unit did not) and silently falls back to 0x14 when the scan fails; its reads
then return False instead of raising. These tests pin the diagnosis logic.
"""
import pytest

from body.nox_i2c_diag import (
    UNIT_PATH,
    find_i2cdetect,
    mcu_diag,
    parse_i2cdetect,
    path_lacks_sbin,
    raw_read_ok,
)

I2CDETECT_0X15 = """\
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- 15 -- -- -- 19 -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
"""

I2CDETECT_EMPTY = """\
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
"""

# What robot_hat's scan sees when the binary is missing from PATH.
SH_NOT_FOUND = "/bin/sh: 1: i2cdetect: not found\n"


def test_parse_finds_mcu_and_imu():
    assert parse_i2cdetect(I2CDETECT_0X15) == [0x15, 0x19]


def test_parse_skips_empty_and_kernel_claimed_cells():
    out = I2CDETECT_0X15.replace("15", "UU")
    assert parse_i2cdetect(out) == [0x19]


def test_parse_of_shell_error_is_empty():
    """The exact failure mode: 'not found' parses to nothing, and the SDK then
    falls back to 0x14 without a word."""
    assert parse_i2cdetect(SH_NOT_FOUND) == []
    assert parse_i2cdetect("") == []


@pytest.mark.parametrize("path", [
    "/home/cat1/.local/bin:/usr/local/bin:/usr/bin:/bin",   # the shipped unit
    "",
    None,
])
def test_old_unit_path_lacks_sbin(path):
    assert path_lacks_sbin(path) is True


@pytest.mark.parametrize("path", [
    UNIT_PATH,
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",  # sudo secure_path
    "/usr/sbin/",
])
def test_fixed_paths_have_sbin(path):
    assert path_lacks_sbin(path) is False


def test_find_i2cdetect_reports_off_path(tmp_path, monkeypatch):
    fake = tmp_path / "i2cdetect"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    on_path, exe = find_i2cdetect(str(tmp_path))
    assert on_path is True and exe == str(fake)
    monkeypatch.setattr("body.nox_i2c_diag.SBIN_DIRS", (str(tmp_path),))
    on_path, exe = find_i2cdetect("/nonexistent")
    assert on_path is False and exe == str(fake), "resolved even when not on PATH"


@pytest.mark.parametrize("raw", [[0x12, 0x34], [0], 7, [255, 0]])
def test_real_bytes_are_ok(raw):
    assert raw_read_ok(raw) is True


@pytest.mark.parametrize("raw", [False, [False, False], [0x12, False], None, [], True, "0"])
def test_robot_hat_false_bytes_are_not_ok(raw):
    """robot_hat returns False per byte instead of raising; (False<<8)+False
    is the 0.0 V that misled issue #12 for two months."""
    assert raw_read_ok(raw) is False


def test_responding_mcu_is_ok_without_scanning():
    diag = mcu_diag(0x14, [0x00, 0x10], path=UNIT_PATH, scan={"addresses": [], "error": "must not be used"})
    assert diag["verdict"] == "ok"
    assert diag["responding"] is True
    assert "bus_devices" not in diag


def test_wrong_address_is_named_with_fix():
    """The issue #12 shape: SDK on 0x14, board on 0x15, unit PATH without sbin."""
    scan = {"addresses": [0x15, 0x19], "error": None}
    diag = mcu_diag(0x14, [False, False], path="/usr/local/bin:/usr/bin:/bin", scan=scan)
    assert diag["verdict"] == "wrong_address"
    assert diag["mcu_addr"] == "0x14"
    assert diag["bus_devices"] == ["0x15", "0x19"]
    assert "0x15" in diag["error"] and "PATH" in diag["error"]
    assert "install-body.sh" in diag["hint"]
    assert diag["path_lacks_sbin"] is True


def test_no_mcu_on_bus():
    scan = {"addresses": [0x19], "error": None}
    diag = mcu_diag(0x14, [False, False], path=UNIT_PATH, scan=scan)
    assert diag["verdict"] == "mcu_absent"
    assert "I2C enabled" in diag["hint"]


def test_scan_failed_when_tool_missing():
    scan = {"addresses": [], "error": "i2cdetect not installed (apt install i2c-tools)"}
    diag = mcu_diag(0x14, False, path=UNIT_PATH, scan=scan)
    assert diag["verdict"] == "scan_failed"
    assert "i2c-tools" in diag["error"]


def test_mcu_present_but_silent():
    scan = {"addresses": [0x14], "error": None}
    diag = mcu_diag(0x14, [False], path=UNIT_PATH, scan=scan)
    assert diag["verdict"] == "unreachable"


def test_unknown_address_still_diagnoses():
    scan = {"addresses": [0x15], "error": None}
    diag = mcu_diag(None, False, path=UNIT_PATH, scan=scan)
    assert diag["mcu_addr"] == "unknown"
    assert diag["verdict"] == "wrong_address"
