"""Behavior tests for the public API of multi_agent_router."""

from __future__ import annotations

import pytest

import multi_agent_router


def test_normalize_trims_and_lowercases() -> None:
    assert multi_agent_router.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        multi_agent_router.normalize(value)


def test_version_is_semver() -> None:
    parts = multi_agent_router.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
