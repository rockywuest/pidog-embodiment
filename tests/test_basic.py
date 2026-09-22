def test_import_body():
    import body
    assert body is not None


def test_import_brain():
    import brain
    assert brain is not None


def test_import_body_helper_modules():
    """The stdlib-only helpers the daemon imports at startup: if one of these
    breaks, nox-body dies on import (issues #12, #24, #25)."""
    import body.nox_audio
    import body.nox_i2c_diag
    import body.nox_motion
    assert body.nox_i2c_diag.MCU_ADDRESSES
    assert body.nox_motion.BUFFER_NAMES
    assert body.nox_audio.AplayMusic


def test_import_body_adapters_base():
    import body.adapters.base
    assert body.adapters.base is not None


def test_import_body_services():
    import body.services
    assert body.services is not None


def test_import_brain_nox_body_client():
    import brain.nox_body_client
    assert brain.nox_body_client is not None