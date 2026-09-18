"""I2C reachability diagnostics for the robot_hat MCU (issue #12).

Every servo, the battery ADC and the IMU sit behind one I2C device: the
robot_hat MCU, at 0x14, 0x15 or 0x16 depending on the board batch. SunFounder's
``robot_hat`` finds the right address by shelling out to ``i2cdetect -y 1`` and
takes the first hit. If that command cannot be found the scan returns nothing
and the library silently falls back to 0x14.

Worse, every I2C read/write in robot_hat is wrapped in a retry that swallows
``OSError`` and returns ``False``. Two ``False`` bytes combine to a battery of
exactly 0.0 V, servo writes "succeed", and the IMU init just prints ``fail``.
Nothing raises. A daemon that talks to the wrong address therefore looks
perfectly healthy while the dog never moves.

``i2cdetect`` lives in ``/usr/sbin``. ``sudo`` puts that on PATH (which is why
SunFounder's examples always work); a systemd unit with its own ``PATH=`` may
not. This module has no SDK imports so it can be unit-tested anywhere; the
daemon feeds it the live probe result.
"""
import os
import re
import shutil
import subprocess

MCU_ADDRESSES = (0x14, 0x15, 0x16)
SBIN_DIRS = ("/usr/sbin", "/sbin", "/usr/local/sbin")
I2C_BUS = 1

# What a unit needs so robot_hat's scan works. install-body.sh rewrites the
# ~/.local/bin part; the sbin dirs are what issue #12 was missing.
UNIT_PATH = "/home/pidog/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def path_lacks_sbin(path):
    """True when none of the sbin directories is on the given PATH string."""
    parts = [p for p in (path or "").split(os.pathsep) if p]
    return not any(p.rstrip("/") in SBIN_DIRS for p in parts)


def find_i2cdetect(path=None):
    """(on_path, resolved_path). Resolved even when it is NOT on PATH, so the
    diagnostics can still run the scan the SDK could not."""
    env_path = os.environ.get("PATH", "") if path is None else path
    on_path = shutil.which("i2cdetect", path=env_path)
    if on_path:
        return True, on_path
    for d in SBIN_DIRS:
        candidate = os.path.join(d, "i2cdetect")
        if os.access(candidate, os.X_OK):
            return False, candidate
    return False, None


def parse_i2cdetect(output):
    """Addresses from ``i2cdetect -y N`` output. Mirrors robot_hat's parser,
    including its blind spots: ``--`` is empty, ``UU`` is claimed by a kernel
    driver, and the header line is dropped."""
    addresses = []
    for line in (output or "").splitlines()[1:]:
        if ":" not in line:
            continue
        row = line.split(":", 1)[1]
        for cell in row.split():
            if re.fullmatch(r"[0-9a-fA-F]{2}", cell):
                addresses.append(int(cell, 16))
    return addresses


def scan_bus(bus=I2C_BUS, path=None, timeout=5):
    """Run i2cdetect ourselves. Returns {"addresses": [...], "i2cdetect": path,
    "i2cdetect_on_path": bool, "error": str|None}."""
    on_path, exe = find_i2cdetect(path)
    result = {"addresses": [], "i2cdetect": exe, "i2cdetect_on_path": on_path, "error": None}
    if not exe:
        result["error"] = "i2cdetect not installed (apt install i2c-tools)"
        return result
    try:
        out = subprocess.run([exe, "-y", str(bus)], capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            result["error"] = (out.stderr or out.stdout).strip() or f"i2cdetect exit {out.returncode}"
        result["addresses"] = parse_i2cdetect(out.stdout)
    except Exception as e:  # noqa: BLE001 - diagnostics must never raise
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def raw_read_ok(raw):
    """True only when every byte of a robot_hat read is a real int. robot_hat
    returns ``False`` (not an exception) for a byte it could not read."""
    if raw is None or isinstance(raw, bool):
        return False
    if isinstance(raw, int):
        return True
    try:
        items = list(raw)
    except TypeError:
        return False
    return bool(items) and all(isinstance(b, int) and not isinstance(b, bool) for b in items)


def mcu_diag(resolved_addr, raw, path=None, scan=None):
    """Turn a live probe into a verdict with a fix hint.

    resolved_addr: the address robot_hat chose (int) or None if unknown.
    raw:           result of a robot_hat I2C read against that address.
    path:          the PATH of the daemon process (default: os.environ).
    scan:          optional pre-computed scan_bus() result; only computed when
                   the probe failed, because it spawns a subprocess.
    """
    env_path = os.environ.get("PATH", "") if path is None else path
    addr_s = f"0x{resolved_addr:02X}" if isinstance(resolved_addr, int) else "unknown"
    diag = {
        "mcu_addr": addr_s,
        "responding": raw_read_ok(raw),
        "i2cdetect_on_path": find_i2cdetect(env_path)[0],
        "path_lacks_sbin": path_lacks_sbin(env_path),
    }
    if diag["responding"]:
        diag["verdict"] = "ok"
        return diag

    if scan is None:
        scan = scan_bus(path=env_path)
    diag["bus_devices"] = [f"0x{a:02X}" for a in scan.get("addresses", [])]
    diag["scan_error"] = scan.get("error")
    mcus_on_bus = [a for a in scan.get("addresses", []) if a in MCU_ADDRESSES]

    if mcus_on_bus and resolved_addr not in mcus_on_bus:
        real = f"0x{mcus_on_bus[0]:02X}"
        diag["verdict"] = "wrong_address"
        diag["error"] = (f"robot_hat talks to {addr_s} but the MCU answers on {real}: "
                         "i2cdetect was not on this process's PATH, so the SDK's bus scan "
                         f"found nothing and fell back to {addr_s}. Every servo write, the "
                         "battery ADC and the IMU fail silently.")
        diag["hint"] = ("put /usr/sbin on the service PATH: git pull && "
                        "sudo ./scripts/install-body.sh && sudo systemctl restart nox-body")
    elif scan.get("error") and not scan.get("addresses"):
        diag["verdict"] = "scan_failed"
        diag["error"] = f"MCU at {addr_s} not responding and the bus scan failed: {scan['error']}"
        diag["hint"] = "run: i2cdetect -y 1   (sudo apt install i2c-tools if missing)"
    elif not mcus_on_bus:
        diag["verdict"] = "mcu_absent"
        diag["error"] = (f"no robot_hat MCU on I2C bus {I2C_BUS} (looked for 0x14/0x15/0x16, "
                         f"found {diag['bus_devices'] or 'nothing'})")
        diag["hint"] = ("robot_hat power switch ON? hat seated? I2C enabled "
                        "(sudo raspi-config → Interface Options → I2C)?")
    else:
        diag["verdict"] = "unreachable"
        diag["error"] = (f"MCU at {addr_s} is on the bus but does not answer reads: "
                         "another process may hold it, or the hat is mid-reset")
        diag["hint"] = "sudo systemctl restart nox-body; if it persists, power-cycle the hat"
    return diag
