"""Test basic import and module structure."""

import pytest


def test_import_boosters():
    """Test that we can import the boosters module."""
    import boosters
    assert hasattr(boosters, '__version__')
    assert hasattr(boosters, 'GBDTBooster')
    assert hasattr(boosters, 'GBLinearBooster')


def test_version_format():
    """Test that version is a valid semver string."""
    from boosters import __version__
    parts = __version__.split('.')
    assert len(parts) >= 2, "Version should have at least major.minor"
    # Check that parts are numeric (or last part may have suffix like '0-dev')
    assert parts[0].isdigit(), "Major version should be numeric"
    assert parts[1].isdigit(), "Minor version should be numeric"


def test_exports():
    """Test that __all__ contains expected exports."""
    import boosters
    expected = {'__version__', 'GBDTBooster', 'GBLinearBooster'}
    actual = set(boosters.__all__)
    assert expected == actual, f"Unexpected exports: {actual - expected}"
