"""Pre-execution filtering logic for AI symbol evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from ..core.position import Position
from ..utils import ensure_float, utc_now

try:  # pragma: no cover - hint-only imports
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from ..core.interfaces import AccountSnapshot, SymbolData


class PreExecutionFilter:
    """Applies deterministic filters before invoking the AI model."""

    def __init__(
        self,
        *,
        cooldown_bars: int = 0,
        long_only: bool = False,
        max_active_positions: Optional[int] = None,
        min_reduce_notional: float = 0.0,
    ) -> None:
        self._cooldown_bars = max(int(cooldown_bars or 0), 0)
        self._long_only = bool(long_only)
        self._max_active_positions = max_active_positions if max_active_positions else None
        self._min_reduce_notional = float(min_reduce_notional or 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_skip_ai(
        self,
        symbol: str,
        data: "SymbolData",
        account_snapshot: "AccountSnapshot",
        position: Optional[Position],
    ) -> Tuple[bool, str]:
        """Return (should_skip, reason) for a symbol before AI invocation."""

        if self._max_active_positions is not None and not self._has_position(position):
            active_positions = self._count_active_positions(account_snapshot)
            if active_positions >= self._max_active_positions:
                return True, "max_positions"

        if self._cooldown_bars and self._in_cooldown(data, position):
            return True, "cooldown"

        return False, ""

    def get_allowed_actions(
        self,
        symbol: str,
        position: Optional[Position],
        account_summary: Dict[str, Any],
    ) -> Iterable[str]:
        """Return a set of allowed actions for the current symbol state."""
        risk_mode = str(account_summary.get("risk_mode") or "").lower()
        if risk_mode in {"halt", "panic", "freeze"}:
            return {"HOLD", "REDUCE", "CLOSE"}

        allowed = {"HOLD"}
        has_position = self._has_position(position)

        reached_risk_cap = False
        if not has_position:
            if position:
                position.reached_risk_cap = False
            allowed.add("WAIT")
        else:
            risk_unit = ensure_float(account_summary.get("risk_unit_R"), 0.0)
            risk_limit = ensure_float(account_summary.get("max_risk_total_R"), 0.0)
            current_risk = self._estimate_position_risk(position)

            reached_risk_cap = current_risk == float("inf")
            if risk_unit > 0 and not reached_risk_cap:
                if risk_limit > 0:
                    threshold = risk_limit - risk_unit
                    if threshold <= 0:
                        reached_risk_cap = True
                    else:
                        reached_risk_cap = not (current_risk < threshold)
                else:
                    reached_risk_cap = False
            elif risk_unit <= 0 and not reached_risk_cap:
                reached_risk_cap = False

            if position:
                position.reached_risk_cap = reached_risk_cap

            if not reached_risk_cap:
                allowed.add("ADD")

        reduce_allowed = has_position
        if reduce_allowed and self._min_reduce_notional > 0 and position is not None:
            amount_val = ensure_float(getattr(position, "amount", 0.0), 0.0)
            price_val = ensure_float(getattr(position, "mark", None), None)
            if price_val is None or price_val <= 0:
                price_val = ensure_float(getattr(position, "entry", 0.0), 0.0)
            multiplier = ensure_float(getattr(position, "multiplier", None), 1.0)
            if multiplier <= 0:
                multiplier = 1.0
            notional = abs(amount_val) * price_val * multiplier if price_val > 0 else 0.0
            if notional < self._min_reduce_notional:
                reduce_allowed = False

        action_block = {
            "ADJUST_SL",
            "ADJUST_TP",
            "TRAIL_ON",
            "TRAIL_OFF",
            "CLOSE",
        }
        if reduce_allowed:
            action_block.update({"REDUCE", "PARTIAL_TP"})
        allowed.update(action_block)
        allowed.add("BUY")
        if not self._long_only:
            allowed.add("SELL")
        return allowed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_position(position: Optional[Position]) -> bool:
        if not position:
            return False
        try:
            return abs(float(position.amount)) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _estimate_position_risk(position: Position) -> float:
        """Return an approximate risk exposure for the position in account currency."""

        if not position:
            return 0.0

        amount = ensure_float(getattr(position, "amount", 0.0), 0.0)
        if abs(amount) <= 0:
            return 0.0

        stop_loss = ensure_float(getattr(position, "sl", None), None)
        if stop_loss is None or stop_loss <= 0:
            return float("inf")

        entry_price = ensure_float(getattr(position, "entry", None), None)
        mark_price = ensure_float(getattr(position, "mark", None), None)
        reference_price = entry_price if entry_price and entry_price > 0 else mark_price
        if reference_price is None or reference_price <= 0:
            return float("inf")

        multiplier = ensure_float(getattr(position, "multiplier", None), 1.0)
        if multiplier <= 0:
            multiplier = 1.0

        risk_per_unit = abs(reference_price - stop_loss) * multiplier
        if risk_per_unit <= 0:
            return 0.0

        return abs(amount) * risk_per_unit

    @staticmethod
    def _count_active_positions(account_snapshot: "AccountSnapshot") -> int:
        raw_positions = {}
        if hasattr(account_snapshot, "raw") and isinstance(account_snapshot.raw, dict):
            raw_positions = account_snapshot.raw
        positions: Sequence[Dict[str, Any]] = raw_positions.get("positions") or []  # type: ignore[assignment]
        active = 0
        for entry in positions:
            if not isinstance(entry, dict):
                continue
            amount = ensure_float(entry.get("contracts") or entry.get("amount") or entry.get("size"), 0.0)
            if abs(amount) > 0:
                active += 1
        return active

    def _in_cooldown(self, data: "SymbolData", position: Optional[Position]) -> bool:
        if self._cooldown_bars <= 0:
            return False
        last_exit_raw = data.meta.get("last_exit_ts") if isinstance(data.meta, dict) else None
        if last_exit_raw is None:
            return False
        try:
            last_exit_ts = float(last_exit_raw)
        except (TypeError, ValueError):
            return False
        current_ts = getattr(data, "last_ts", 0) or 0
        if current_ts > 1e11:
            current_ts /= 1000
        if not current_ts:
            try:
                current_ts = float(data.meta.get("current_ts"))  # type: ignore[assignment]
            except Exception:
                current_ts = utc_now().timestamp()
        interval = 0.0
        if isinstance(data.meta, dict):
            interval = ensure_float(data.meta.get("suggested_interval_seconds"), 0.0)
            if interval <= 0:
                timeframes_meta = data.meta.get("timeframes")
                if isinstance(timeframes_meta, dict):
                    for seconds in timeframes_meta.values():
                        interval = ensure_float(seconds, 0.0)
                        if interval > 0:
                            break
        if interval <= 0:
            interval = 3600.0
        delta = current_ts - last_exit_ts
        if delta < 0:
            delta = 0.0
        bars_since = delta / interval if interval else float("inf")
        return bars_since < self._cooldown_bars


__all__ = ["PreExecutionFilter"]
