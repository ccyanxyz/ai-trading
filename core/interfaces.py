"""Core interfaces and data structures for AI trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, Sequence

from .position import Position


class MarketAdapter(Protocol):
    """Adapter interface that bridges a specific market/exchange into the AI engine."""

    def normalize_symbol(self, symbol: str) -> str:
        ...

    async def setup(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def fetch_account_state(self) -> Dict[str, Any]:
        ...

    async def collect_symbol_data(
        self,
        symbol: str,
        account_state: Dict[str, Any],
        *,
        position: Optional[Position],
        risk: Dict[str, Any],
        constraints: Dict[str, Any],
        use_chart: bool = True,
    ) -> Optional["SymbolData"]:
        ...

    async def execute_decision(
        self,
        symbol: str,
        data: "SymbolData",
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        ...


@dataclass
class SymbolData:
    symbol: str
    payload: Dict[str, Any]
    last_ts: int
    last_price: float
    meta: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Position] = None


@dataclass
class AccountSnapshot:
    """Normalized view of the account state shared with downstream components."""

    raw: Dict[str, Any]
    prompt_payload: Dict[str, Any]
    summary: Dict[str, Any]
    positions: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolSchedule:
    """Scheduling metadata for a symbol."""

    symbol: str
    interval_seconds: Optional[int]
    next_due_ts: Optional[float] = None
    last_processed_ts: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccountFormatter(Protocol):
    def format(self, raw: Dict[str, Any]) -> AccountSnapshot:
        ...


class SchedulePolicy(Protocol):
    def prepare(self, strategy: "AITradingStrategy") -> None:
        ...

    def ensure_symbol(self, strategy: "AITradingStrategy", symbol: str) -> None:
        ...

    def drop_symbol(self, strategy: "AITradingStrategy", symbol: str) -> None:
        ...

    def should_process(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        data: SymbolData,
        last_processed_ts: int,
        *,
        force: bool,
    ) -> bool:
        ...

    def record_success(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        data: SymbolData,
        decision: Optional[Dict[str, Any]],
    ) -> None:
        ...

    def next_run_delay(
        self,
        strategy: "AITradingStrategy",
        last_run_time: Optional[datetime],
    ) -> float:
        ...


class BatchPlanner(Protocol):
    def plan_batches(
        self,
        strategy: "AITradingStrategy",
        symbol_data: Sequence[SymbolData],
    ) -> Sequence[Sequence[SymbolData]]:
        ...


class PayloadBuilder(Protocol):
    def build_payload(
        self,
        strategy: "AITradingStrategy",
        timestamp_iso: str,
        account: AccountSnapshot,
        batch: Sequence[SymbolData],
    ) -> Dict[str, Any]:
        ...


class TradePlanHandler(Protocol):
    async def update_plan(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        decision: Dict[str, Any],
    ) -> None:
        ...

    def record_execution_history(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        decision: Dict[str, Any],
        order_result: Dict[str, Any],
    ) -> None:
        ...

    def archive_trade(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
    ) -> None:
        ...


class ExecutionAdapter(Protocol):
    async def execute(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        data: SymbolData,
        decision: Dict[str, Any],
        account: AccountSnapshot,
    ) -> Optional[Dict[str, Any]]:
        ...


class AIEnvelopeAdapter(Protocol):
    """Transforms between local strategy state and AI-facing payloads."""

    def encode_request(
        self,
        strategy: "AITradingStrategy",
        payload: Dict[str, Any],
        *,
        account: AccountSnapshot,
        symbols: Sequence[SymbolData],
    ) -> Dict[str, Any]:
        ...

    def decode_response(
        self,
        strategy: "AITradingStrategy",
        response: Any,
        *,
        payload: Dict[str, Any],
        account: AccountSnapshot,
        symbols: Sequence[SymbolData],
    ) -> Any:
        ...


__all__ = [
    "MarketAdapter",
    "SymbolData",
    "AccountSnapshot",
    "SymbolSchedule",
    "AccountFormatter",
    "SchedulePolicy",
    "BatchPlanner",
    "PayloadBuilder",
    "TradePlanHandler",
    "ExecutionAdapter",
    "AIEnvelopeAdapter",
]
