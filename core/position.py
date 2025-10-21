"""Position-centric data structures for AI trading strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    """Unified representation of an AI-managed trading position."""

    symbol: str
    side: str
    entry: float
    amount: float
    upnl: float
    mark: float
    timestamp: str

    playbook: Optional[str] = None
    confidence: Optional[float] = None
    rr: Optional[float] = None
    reason: Optional[str] = None
    sl: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    tp_policy: Optional[str] = None
    tp_lock: Optional[Dict[str, float]] = None
    trailing: Optional[Dict[str, Any]] = None
    actual_rr: Optional[float] = None
    triggered_targets: List[float] = field(default_factory=list)
    trail_state: Dict[str, Any] = field(default_factory=dict)
    revision: int = 0

    multiplier: Optional[float] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    reached_risk_cap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the position."""

        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Restore a position from persisted payload."""

        allowed = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**allowed)


__all__ = ["Position"]
