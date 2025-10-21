"""Shared strategy execution context objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .interfaces import AccountSnapshot
from .position import Position

if TYPE_CHECKING:  # pragma: no cover
    from ..services.position_manager import PositionManager


@dataclass
class StrategyContext:
    """Execution context shared between strategy controller and pipeline."""

    strategy_key: str
    config: Dict[str, Any]
    symbols: List[str]
    positions: Dict[str, Position]
    last_processed: Dict[str, int]
    account_snapshot: Optional[AccountSnapshot] = None
    latest_decisions: List[Dict[str, Any]] = field(default_factory=list)
    last_run_time: Optional[datetime] = None
    last_error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    position_manager: Optional["PositionManager"] = None

    def copy(self) -> "StrategyContext":
        """Create a shallow copy of the context for safe mutation."""

        return StrategyContext(
            strategy_key=self.strategy_key,
            config=self.config,
            symbols=list(self.symbols),
            positions=self.positions,
            last_processed=self.last_processed.copy(),
            account_snapshot=self.account_snapshot,
            latest_decisions=list(self.latest_decisions),
            last_run_time=self.last_run_time,
            last_error=self.last_error,
            extras=dict(self.extras),
            position_manager=self.position_manager,
        )


__all__ = ["StrategyContext"]
