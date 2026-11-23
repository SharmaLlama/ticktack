"""Public API surface snapshot guard for ``ticktack.__all__`` (Section 3 Gate 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ticktack  # noqa: E402

SNAPSHOT_PATH = Path(__file__).resolve().parent / "public_surface_snapshot.json"


def test_public_surface_matches_snapshot() -> None:
    """Fail if ``ticktack.__all__`` changes without updating ``public_surface_snapshot.json``."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    expected = snapshot["all"]
    actual = sorted(ticktack.__all__)

    assert len(ticktack.__all__) == len(set(ticktack.__all__)), (
        "Duplicate entries found in ticktack.__all__."
    )
    assert expected == sorted(set(expected)), (
        "tests/public_surface_snapshot.json must store a sorted, deduplicated list."
    )

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    assert actual == expected, (
        "ticktack.__all__ diverged from tests/public_surface_snapshot.json.\n"
        f"  Added ({len(added)}): {added}\n"
        f"  Removed ({len(removed)}): {removed}\n"
        "Every new public name requires a docs example, a test, and an explicit snapshot update."
    )


def test_public_surface_exports_resolve() -> None:
    """Every name listed in ``ticktack.__all__`` must resolve on the top-level package."""
    missing = [name for name in ticktack.__all__ if not hasattr(ticktack, name)]
    assert missing == [], f"Unresolved names in ticktack.__all__: {missing}"
