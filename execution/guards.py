"""Execution guard helpers to enforce deterministic risk constraints."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import logging
import math

from ..core.position import Position
from ..utils import ensure_float, first_float

try:  # pragma: no cover - typing helpers
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from ..core.interfaces import AccountSnapshot, SymbolData


class ExecutionGuard:
    """Validates AI decisions and derives safe execution sizes."""

    def __init__(
        self,
        *,
        min_rr: float = 0.0,
        long_only: bool = False,
        min_reduce_notional: float = 0.0,
    ) -> None:
        self._min_rr = float(min_rr or 0.0)
        self._long_only = bool(long_only)
        self._min_reduce_notional = float(min_reduce_notional or 0.0)
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        account_snapshot: "AccountSnapshot",
        position: Optional[Position],
        allowed_actions: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, str]:
        action = str(decision.get("action") or decision.get("act") or "").upper()
        if not action:
            return False, "missing_action"

        account_summary = getattr(account_snapshot, "summary", None)

        if allowed_actions:
            allowed_set = {a.upper() for a in allowed_actions}
            self._logger.info(
                "Allowed actions check: symbol=%s action=%s allowed=%s summary=%s",
                symbol,
                action,
                sorted(allowed_set),
                account_summary,
            )
            if action not in allowed_set:
                return False, "action_not_allowed"
        else:
            self._logger.info(
                "Allowed actions not provided: symbol=%s action=%s summary=%s",
                symbol,
                action,
                account_summary,
            )

        if self._long_only and action == "SELL" and not position:
            return False, "long_only"

        if action in {"BUY", "ADD"}:
            rr_value = first_float(decision, ("rr", "rr_after_costs"))
            if self._min_rr and (rr_value is None or rr_value < self._min_rr):
                return False, "rr_too_low"
            if decision.get("sl") is None:
                return False, "missing_sl"

        if action in {"REDUCE", "PARTIAL_TP"}:
            if position is None:
                return False, "no_position"
            if self._min_reduce_notional > 0:
                qty = ensure_float(getattr(position, "amount", 0.0), 0.0)
                price = ensure_float(getattr(position, "mark", None), None)
                if price is None or price <= 0:
                    price = ensure_float(getattr(position, "entry", 0.0), 0.0)
                multiplier = ensure_float(getattr(position, "multiplier", None), 1.0)
                if multiplier <= 0:
                    multiplier = 1.0
                notional = abs(qty) * price * multiplier
                if notional < self._min_reduce_notional:
                    return False, "notional_too_small"

        return True, "ok"

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------

    def calc_position_size(
        self,
        decision: Dict[str, Any],
        account_snapshot: "AccountSnapshot",
        symbol_data: "SymbolData",
        position: Optional[Position] = None,
    ) -> float:
        """Compute position size using the Kelly criterion with safety checks."""

        action = str(decision.get("action") or decision.get("act") or "").upper()
        if action in {"REDUCE", "PARTIAL_TP"}:
            return position.amount * 0.3
        
        if action not in {"BUY", "SELL", "ADD"}:
            return 0.0

        summary = account_snapshot.summary if hasattr(account_snapshot, "summary") else {}
        risk_budget = first_float(summary, ("risk_unit_R", "risk_unit_r"))
        if risk_budget is None or risk_budget <= 0:
            self._logger.warning("Risk budget not available for sizing: %s", risk_budget)
            return 0.0

        win_prob = first_float(decision, ("confidence", "conf"))
        rr_value = first_float(decision, ("rr", "rr_after_costs"))
        if win_prob is None or rr_value is None:
            self._logger.warning(
                "Missing confidence or RR for %s decision: conf=%s rr=%s",
                action,
                win_prob,
                rr_value,
            )
            return 0.0

        win_prob = max(0.01, min(win_prob, 0.99))
        rr_value = max(rr_value, 0.1)

        kelly_fraction = (win_prob * (rr_value + 1) - 1) / rr_value
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
        if kelly_fraction <= 0:
            self._logger.info(
                "Kelly fraction <= 0, skip sizing: conf=%.4f rr=%.4f", win_prob, rr_value
            )
            return 0.0

        last_price = ensure_float(getattr(symbol_data, "last_price", 0.0), 0.0)
        if last_price <= 0:
            last_price = ensure_float(decision.get("price") or decision.get("entry"), 0.0)
        sl_value = ensure_float(decision.get("sl"), 0.0)
        if last_price <= 0 or sl_value <= 0:
            self._logger.warning(
                "Invalid price/sl for sizing: price=%s sl=%s", last_price, sl_value
            )
            return 0.0

        risk_per_unit = abs(last_price - sl_value)
        if risk_per_unit <= 0:
            self._logger.warning(
                "Stop distance <= 0 for sizing: price=%s sl=%s", last_price, sl_value
            )
            return 0.0

        multiplier = 1.0
        meta = getattr(symbol_data, "meta", {}) if hasattr(symbol_data, "meta") else {}
        if isinstance(meta, dict):
            multiplier = ensure_float(meta.get("multiplier"), 1.0)
        if multiplier <= 0:
            multiplier = 1.0

        risk_per_contract = risk_per_unit * multiplier
        if risk_per_contract <= 0:
            self._logger.warning(
                "Risk per contract <= 0 after multiplier: price=%s sl=%s multiplier=%s",
                last_price,
                sl_value,
                multiplier,
            )
            return 0.0

        risk_capital = kelly_fraction * risk_budget
        qty = risk_capital / risk_per_contract

        bounds = getattr(symbol_data, "meta", {}).get("bounds") if hasattr(symbol_data, "meta") else None
        lot_step = 0.0
        if isinstance(bounds, dict):
            qty_abs = bounds.get("qty_abs") or []
            min_qty = ensure_float(qty_abs[0], 0.0) if len(qty_abs) > 0 else 0.0
            max_qty = ensure_float(qty_abs[1], float("inf")) if len(qty_abs) > 1 else float("inf")
            lot_step = ensure_float(bounds.get("lot_step"), 0.0)
            if qty < min_qty:
                self._logger.warning(
                    "Sized quantity %.6f < exchange minimum %.6f, skip position", qty, min_qty
                )
                return 0.0
            qty = min(qty, max_qty)

        if lot_step > 0:
            qty = math.floor(qty / lot_step + 1e-9) * lot_step
        qty = max(qty, 0.0)
        if lot_step > 0 and qty < lot_step:
            self._logger.warning("Rounded quantity below minimum lot step %.6f", lot_step)
            return 0.0

        self._logger.info(
            "Sizing via Kelly: action=%s conf=%.4f rr=%.4f kelly=%.2f%% risk=%.2f stop=%.4f multiplier=%.2f risk_per_contract=%.4f qty=%.6f",
            action,
            win_prob,
            rr_value,
            kelly_fraction * 100,
            risk_capital,
            risk_per_unit,
            multiplier,
            risk_per_contract,
            qty,
        )

        return qty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

__all__ = ["ExecutionGuard"]
