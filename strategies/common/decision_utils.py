"""Shared helpers for normalizing decision payloads across strategies."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...utils import coerce_optional_float

DEFAULT_DECISION_KEY_ALIASES: Dict[str, str] = {
    "sym": "symbol",
    "act": "action",
    "amt": "amount",
    "pb": "playbook",
    "rsn": "reason",
    "conf": "confidence",
    "rr": "rr",
    "sl": "sl",
    "tgts": "targets",
    "tp_pol": "tp_policy",
    "tp_lock": "tp_lock",
    "trail": "trailing",
}

REVERSE_DECISION_ALIASES: Dict[str, str] = {
    canonical: alias
    for alias, canonical in DEFAULT_DECISION_KEY_ALIASES.items()
    if alias != canonical
}


def rename_decision_keys(
    payload: Dict[str, Any],
    aliases: Dict[str, str] = DEFAULT_DECISION_KEY_ALIASES,
) -> Dict[str, Any]:
    renamed: Dict[str, Any] = {}
    for key, value in payload.items():
        canonical = aliases.get(key, key)
        if canonical in renamed and canonical != key:
            continue
        renamed[canonical] = value
    return renamed


def normalize_decision_payload(
    entry: Dict[str, Any],
    *,
    aliases: Dict[str, str] = DEFAULT_DECISION_KEY_ALIASES,
    symbol_formatter: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    renamed = rename_decision_keys(entry, aliases)
    symbol = renamed.get("symbol")
    if not symbol:
        return None
    symbol_text = str(symbol)
    renamed["symbol"] = symbol_formatter(symbol_text) if callable(symbol_formatter) else symbol_text

    action = renamed.get("action")
    if action is not None:
        renamed["action"] = str(action).upper()

    for field in ("amount", "confidence", "rr"):
        if field in renamed:
            renamed[field] = coerce_optional_float(renamed[field])

    targets = renamed.get("targets")
    if isinstance(targets, list):
        renamed["targets"] = [coerce_optional_float(value) for value in targets]

    return dict(renamed)


__all__ = [
    "DEFAULT_DECISION_KEY_ALIASES",
    "REVERSE_DECISION_ALIASES",
    "rename_decision_keys",
    "normalize_decision_payload",
]
