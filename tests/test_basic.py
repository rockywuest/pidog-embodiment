def test_import_body():
    import body
    assert body is not None


def test_import_brain():
    import brain
    assert brain is not None


def test_import_body_daemon_patch():
    import body.daemon_patch
    assert body.daemon_patch is not None


def test_import_body_adapters_base():
    import body.adapters.base
    assert body.adapters.base is not None


def test_import_body_services():
    import body.services
    assert body.services is not None


def test_import_brain_nox_body_client():
    import brain.nox_body_client
    assert brain.nox_body_client is not None