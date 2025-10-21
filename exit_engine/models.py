"""Data models shared across the exit engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core.position import Position


@dataclass(frozen=True)
class ExitContext:
    """Snapshot passed to exit rules for evaluation."""

    symbol: str
    position: Position
    price: float
    timestamp: float
    account_summary: Dict[str, Any] = field(default_factory=dict)
    symbol_meta: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitDecision:
    """Structured payload describing an automated exit action."""

    symbol: str
    action: str
    amount: Optional[float] = None
    sl: Optional[float] = None
    targets: Optional[list[float]] = None
    tp_policy: Optional[str] = None
    tp_lock: Optional[Dict[str, Any]] = None
    trailing: Optional[Dict[str, Any]] = None
    reason: str = "auto-triggered rule-based exit"
    rule: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_decision_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "sym": self.symbol,
            "action": self.action,
            "reason": self.reason,
            "auto": True,
            "rule": self.rule,
        }
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.sl is not None:
            payload["sl"] = self.sl
        if self.targets is not None:
            payload["targets"] = list(self.targets)
        if self.tp_policy is not None:
            payload["tp_policy"] = self.tp_policy
        if self.tp_lock is not None:
            payload["tp_lock"] = dict(self.tp_lock)
        if self.trailing is not None:
            payload["trailing"] = dict(self.trailing)
        if self.metadata:
            payload["meta"] = dict(self.metadata)
        return payload


__all__ = ["ExitContext", "ExitDecision"]
