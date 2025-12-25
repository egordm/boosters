"""Basic import tests for boosters package."""

import boosters


def test_version():
    """Test that version is exported."""
    assert hasattr(boosters, "__version__")
    assert isinstance(boosters.__version__, str)


def test_import_native_module():
    """Test that native module can be imported."""
    from boosters import _boosters_rs

    assert hasattr(_boosters_rs, "__version__")
