"""Beat categories used by the first research benchmark."""

from __future__ import annotations

from typing import Literal

BeatGroup = Literal["N", "S", "V"]
CLASS_ORDER: tuple[BeatGroup, ...] = ("N", "S", "V")

BEAT_GROUPS: dict[str, BeatGroup] = {
    **dict.fromkeys(("N", "L", "R", "e", "j"), "N"),
    **dict.fromkeys(("A", "a", "J", "S"), "S"),
    **dict.fromkeys(("V", "E"), "V"),
}

OUT_OF_SCOPE_BEATS = frozenset({"F", "/", "f", "Q"})
ALL_BEAT_SYMBOLS = frozenset(BEAT_GROUPS) | OUT_OF_SCOPE_BEATS


def group_beat(symbol: str) -> BeatGroup | None:
    """Map a documented MIT-BIH beat symbol to the first-release category."""
    return BEAT_GROUPS.get(symbol)


def is_beat_annotation(symbol: str) -> bool:
    """Exclude rhythm changes, quality markers, and comments from beat counts."""
    return symbol in ALL_BEAT_SYMBOLS
