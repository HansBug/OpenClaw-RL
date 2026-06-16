from __future__ import annotations

import os
from typing import Any


_DEFAULT_FORCE_TEXT_FIRST_MESSAGE = (
    "Let me quickly confirm one detail before I make changes."
)


def _env_enabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def forced_text_first_message(
    *,
    data_source: Any,
    conversation_mode: Any,
    turn_idx: int,
) -> str | None:
    if not _env_enabled("TAU2_FORCE_TEXT_FIRST"):
        return None
    if str(data_source or "").strip() != "tau2":
        return None
    if str(conversation_mode or "").strip() != "non_solo":
        return None
    if turn_idx != 0:
        return None
    return str(
        os.getenv(
            "TAU2_FORCE_TEXT_FIRST_MESSAGE",
            _DEFAULT_FORCE_TEXT_FIRST_MESSAGE,
        )
    ).strip()
