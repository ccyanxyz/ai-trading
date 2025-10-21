"""Default implementations for AI strategy runtime collaborators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from ..services.scheduler import DefaultSchedulePolicy  # re-export for backwards compatibility
from .interfaces import AccountFormatter, AccountSnapshot, AIEnvelopeAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.strategy import AITradingStrategy
    from .interfaces import SymbolData


def strip_private_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: strip_private_keys(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [strip_private_keys(item) for item in data]
    return data


class DefaultAccountFormatter(AccountFormatter):
    def format(self, raw: Dict[str, Any]) -> AccountSnapshot:
        prompt_payload = strip_private_keys(raw)
        if isinstance(prompt_payload, dict):
            prompt_payload.pop("positions", None)
            account_section = prompt_payload.get("account")
            if isinstance(account_section, dict):
                account_section.pop("positions", None)
        summary_candidates = [
            "net_value",
            "equity",
            "balance",
            "cash",
            "margin_free",
            "available",
            "margin_used",
            "margin",
            "total_pos_notional_value",
        ]
        summary = {key: raw[key] for key in summary_candidates if key in raw}
        if not summary:
            account_info = raw.get("account")
            if isinstance(account_info, dict):
                summary = {
                    key: account_info[key]
                    for key in summary_candidates
                    if key in account_info and key not in summary
                }
        return AccountSnapshot(
            raw=raw,
            prompt_payload=prompt_payload,
            summary=summary or prompt_payload,
            positions=None,
            metadata={},
        )


class DefaultAIEnvelopeAdapter(AIEnvelopeAdapter):
    def encode_request(
        self,
        strategy: "AITradingStrategy",
        payload: Dict[str, Any],
        *,
        account: AccountSnapshot,
        symbols: Sequence["SymbolData"],
    ) -> Dict[str, Any]:
        return payload

    def decode_response(
        self,
        strategy: "AITradingStrategy",
        response: Any,
        *,
        payload: Dict[str, Any],
        account: AccountSnapshot,
        symbols: Sequence["SymbolData"],
    ) -> Any:
        return response


__all__ = [
    "strip_private_keys",
    "DefaultAccountFormatter",
    "DefaultSchedulePolicy",
    "DefaultAIEnvelopeAdapter",
]
