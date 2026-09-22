"""Draining the SDK's motion buffers (issue #25).

SunFounder's ``Pidog`` does not execute an action when you call it. ``do_action``
appends target angles to ``legs_action_buffer`` / ``head_action_buffer`` /
``tail_action_buffer``, and background consumer threads play them out frame by
frame. A long action, or a behaviour engine issuing them in a loop, therefore
leaves a queue behind: issue #25 saw 227 frames still pending.

Stopping the behaviour engine only stops *new* frames from being appended. The
queued ones keep the dog moving for many seconds, and an emergency stop that
merely appends a ``lie`` lands behind all of them — the opposite of stopping.

Clearing the queue is the fix. The SDK's own ``body_stop()`` does it (it calls
``legs_stop()`` / ``head_stop()`` / ``tail_stop()``, each clearing one buffer);
older builds may expose only the per-part methods, and this module falls back to
clearing the lists directly. Kept free of SDK imports so it is testable
anywhere.
"""

BUFFER_NAMES = ("legs_action_buffer", "head_action_buffer", "tail_action_buffer")


def buffer_depth(dog):
    """Total number of motion frames still queued across all three buffers."""
    total = 0
    for name in BUFFER_NAMES:
        buf = getattr(dog, name, None)
        if buf is not None:
            try:
                total += len(buf)
            except TypeError:
                pass
    return total


def _call(dog, name):
    """Call dog.<name>() if it exists. True when it ran without raising."""
    fn = getattr(dog, name, None)
    if not callable(fn):
        return False
    fn()
    return True


def drain(dog):
    """Clear every queued motion frame.

    Returns {"drained": int, "remaining": int, "via": str}, and adds "error"
    when a stop method raised — the caller (an emergency stop, typically) must
    keep going regardless, so nothing propagates.
    """
    before = buffer_depth(dog)
    result = {"via": "none"}
    try:
        if _call(dog, "body_stop"):
            result["via"] = "body_stop"
        elif all(_call(dog, n) for n in ("legs_stop", "head_stop", "tail_stop")):
            result["via"] = "part_stop"
    except Exception as e:  # noqa: BLE001 - never block a stop path
        result["error"] = f"{type(e).__name__}: {e}"

    remaining = buffer_depth(dog)
    if remaining:
        # No usable stop method, or one that did not clear everything: the
        # buffers are plain lists, so empty them ourselves rather than let the
        # dog keep moving.
        for name in BUFFER_NAMES:
            buf = getattr(dog, name, None)
            if buf is None:
                continue
            try:
                buf.clear()
            except AttributeError:
                try:
                    del buf[:]
                except TypeError:
                    pass
        remaining = buffer_depth(dog)
        if result["via"] == "none":
            result["via"] = "clear"
        elif not result.get("error"):
            result["via"] += "+clear"

    result["drained"] = max(0, before - remaining)
    result["remaining"] = remaining
    return result
