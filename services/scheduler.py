"""Scheduling utilities for AI trading strategies."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, TYPE_CHECKING

from ..core.interfaces import SchedulePolicy, SymbolData, SymbolSchedule
from ..utils import utc_now

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.strategy import AITradingStrategy


class DefaultSchedulePolicy(SchedulePolicy):
    """Time-based scheduling with optional per-symbol overrides."""

    def __init__(
        self,
        evaluation_seconds: int,
        poll_delay_seconds: int,
        *,
        symbol_overrides: Optional[Dict[str, int]] = None,
    ) -> None:
        self._evaluation_seconds = evaluation_seconds
        self._poll_delay_seconds = poll_delay_seconds
        self._symbol_overrides = symbol_overrides or {}
        self._schedules: Dict[str, SymbolSchedule] = {}

    def prepare(self, strategy: "AITradingStrategy") -> None:
        for symbol in strategy._symbols:
            self.ensure_symbol(strategy, symbol)

    def ensure_symbol(self, strategy: "AITradingStrategy", symbol: str) -> None:
        if symbol in self._schedules:
            return
        interval = self._symbol_overrides.get(symbol)
        if interval is None:
            interval = self._evaluation_seconds if self._evaluation_seconds > 0 else None
        schedule = SymbolSchedule(symbol=symbol, interval_seconds=interval)
        schedule.next_due_ts = self._aligned_next_due(utc_now().timestamp(), interval)
        self._schedules[symbol] = schedule

    def drop_symbol(self, strategy: "AITradingStrategy", symbol: str) -> None:
        self._schedules.pop(symbol, None)

    def should_process(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        data: SymbolData,
        last_processed_ts: int,
        *,
        force: bool,
    ) -> bool:
        self.ensure_symbol(strategy, symbol)
        if force:
            return True
        schedule = self._schedules[symbol]
        if last_processed_ts and data.last_ts <= last_processed_ts:
            return False
        if schedule.interval_seconds and schedule.next_due_ts is not None:
            now = utc_now().timestamp()
            if now + 1 < schedule.next_due_ts:
                return False
        return True

    def record_success(
        self,
        strategy: "AITradingStrategy",
        symbol: str,
        data: SymbolData,
        decision: Optional[Dict[str, any]],
    ) -> None:
        self.ensure_symbol(strategy, symbol)
        schedule = self._schedules.get(symbol)
        if not schedule:
            return
        suggested_interval = data.meta.get("suggested_interval_seconds") if data.meta else None
        if isinstance(suggested_interval, (int, float)) and suggested_interval > 0:
            schedule.interval_seconds = int(suggested_interval)
        schedule.last_processed_ts = data.last_ts
        schedule.next_due_ts = self._aligned_next_due(
            utc_now().timestamp(), schedule.interval_seconds
        )
        self._schedules[symbol] = schedule

    def next_run_delay(
        self,
        strategy: "AITradingStrategy",
        last_run_time: Optional[datetime],
    ) -> float:
        now = utc_now().timestamp()
        if not self._schedules:
            return max(30.0, float(self._poll_delay_seconds))
        candidates = [
            max(schedule.next_due_ts - now, 0.0)
            for schedule in self._schedules.values()
            if schedule.interval_seconds and schedule.next_due_ts is not None
        ]
        if not candidates:
            return max(30.0, float(self._poll_delay_seconds))
        wait = min(candidates) + float(self._poll_delay_seconds)
        return max(30.0, wait)

    @staticmethod
    def _aligned_next_due(now_ts: float, interval: Optional[int]) -> Optional[float]:
        if not interval or interval <= 0:
            return None
        next_boundary = ((int(now_ts) // interval) + 1) * interval
        return float(next_boundary)


__all__ = ["DefaultSchedulePolicy"]
