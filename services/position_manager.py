"""Position-centric state manager for AI trading strategies."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..core.position import Position
from ..utils import ensure_float, first_float, format_ts, optional_float, utc_now


class PositionManager:
    """Handles in-memory and persisted position state."""

    def __init__(
        self,
        state_store,
        json_safe,
        logger,
    ) -> None:
        self._state_store = state_store
        self._json_safe = json_safe
        self._logger = logger
        self._positions: Dict[str, Position] = {}
        self._last_exits: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        payload = self._state_store.load_positions()
        positions: Dict[str, Position] = {}
        for symbol, data in payload.items():
            try:
                positions[symbol] = Position.from_dict(data)
            except Exception:
                self._logger.exception("Failed to restore position for %s", symbol)
        self._positions = positions

    @staticmethod
    def _extract_side(payload: Dict[str, Any], fallback: Optional[str] = None) -> str:
        side = str(payload.get("side") or fallback or "long").lower()
        if side not in {"long", "short"}:
            return "long"
        return side

    @staticmethod
    def _extract_timestamp(payload: Dict[str, Any]) -> str:
        ts_value = payload.get("timestamp") or payload.get("ts")
        if isinstance(ts_value, (int, float)):
            return format_ts(float(ts_value))
        if isinstance(ts_value, str) and ts_value:
            return ts_value
        return format_ts(utc_now().timestamp())

    @classmethod
    def _coerce_number(cls, payload: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key not in payload:
                continue
            numeric = optional_float(payload.get(key))
            if numeric is not None:
                return numeric
        return None

    @classmethod
    def _extract_fill_price(
        cls,
        order_payload: Dict[str, Any],
        orders_payload: Iterable[Dict[str, Any]],
    ) -> Optional[float]:
        candidate_keys = (
            "avg_price",
            "average",
            "avgPrice",
            "avg",
            "fill_price",
            "fills_avg_price",
        )

        price = cls._coerce_number(order_payload, candidate_keys)
        if price and price > 0:
            return price

        direct_price = cls._coerce_number(order_payload, ("price",))
        if direct_price and direct_price > 0:
            return direct_price

        for item in orders_payload:
            if not isinstance(item, dict):
                continue
            price = cls._coerce_number(item, candidate_keys)
            if price and price > 0:
                return price
            direct_price = cls._coerce_number(item, ("price", "avgPrice"))
            if direct_price and direct_price > 0:
                return direct_price
        return None

    @classmethod
    def _apply_fill_price(
        cls,
        position: Position,
        action: str,
        order_payload: Dict[str, Any],
        orders_payload: Iterable[Dict[str, Any]],
    ) -> None:
        action_upper = (action or "").upper()
        fill_price = cls._extract_fill_price(order_payload, orders_payload)
        if fill_price is None or fill_price <= 0:
            return

        if "BUY" in action_upper and optional_float(position.entry) in (None, 0.0):
            position.entry = fill_price
            if optional_float(position.mark) in (None, 0.0):
                position.mark = fill_price
            cls._update_price_extrema(position, fill_price)
        elif action_upper in {"SELL", "CLOSE", "REDUCE", "PARTIAL_TP"}:
            if optional_float(position.mark) in (None, 0.0):
                position.mark = fill_price
            cls._update_price_extrema(position, fill_price)

    @classmethod
    def _update_price_extrema(cls, position: Position, price: Any) -> None:
        numeric = optional_float(price)
        if numeric is None:
            return
        trail_state = position.trail_state if isinstance(position.trail_state, dict) else {}
        extrema = trail_state.get("price_extrema") if isinstance(trail_state.get("price_extrema"), dict) else {}
        high_val = optional_float(extrema.get("high"))
        if high_val is None or numeric > high_val:
            extrema["high"] = numeric
        low_val = optional_float(extrema.get("low"))
        if low_val is None or numeric < low_val:
            extrema["low"] = numeric
        trail_state["price_extrema"] = extrema
        position.trail_state = trail_state

    @staticmethod
    def _clear_price_extrema(position: Position) -> None:
        if isinstance(position.trail_state, dict):
            position.trail_state.pop("price_extrema", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def last_exit(self, symbol: str) -> Optional[float]:
        return self._last_exits.get(symbol)

    def upsert(
        self,
        symbol: str,
        exchange_position: Optional[Dict[str, Any]],
        decision: Optional[Dict[str, Any]] = None,
        *,
        update_revision: bool = True,
    ) -> Optional[Position]:
        exchange_payload = exchange_position or {}
        position = self._positions.get(symbol)

        entry = first_float(exchange_payload, ("entry_price", "avg_price", "entry"), default=0.0)
        mark = first_float(
            exchange_payload,
            ("mark_price", "mark", "last_price", "price"),
            default=entry,
        )
        amount = first_float(
            exchange_payload,
            ("contracts", "amount", "position_amt", "size"),
            default=0.0,
        )
        upnl = first_float(
            exchange_payload,
            ("unrealized_pnl", "upnl", "unrealized"),
            default=0.0,
        )
        side = self._extract_side(exchange_payload, fallback="long" if amount >= 0 else "short")
        timestamp = self._extract_timestamp(exchange_payload)
        multiplier = optional_float(exchange_payload.get("multiplier")) if exchange_payload else None

        amount_value = amount
        if not position:
            if abs(amount_value) <= 0:
                return None
            position = Position(
                symbol=symbol,
                side=side,
                entry=entry,
                amount=amount,
                upnl=upnl,
                mark=mark,
                timestamp=timestamp,
            )
            if multiplier is not None:
                position.multiplier = multiplier
            self._update_price_extrema(position, mark if mark else entry)
        else:
            position.side = side
            position.entry = entry if entry else position.entry
            position.amount = amount
            position.upnl = upnl
            position.mark = mark
            if not position.timestamp:
                position.timestamp = timestamp
            if multiplier is not None:
                position.multiplier = multiplier

        mark_numeric = optional_float(position.mark)
        amount_numeric = optional_float(position.amount)
        if amount_numeric is not None and abs(amount_numeric) > 0:
            reference_price = mark_numeric if mark_numeric is not None else entry
            self._update_price_extrema(position, reference_price)
        else:
            self._clear_price_extrema(position)

        self._apply_decision_metadata(position, decision)
        try:
            if abs(float(position.amount)) > 0:
                self._last_exits.pop(symbol, None)
        except (TypeError, ValueError):
            pass

        if update_revision:
            position.revision += 1
        self._positions[symbol] = position
        return position

    def update_risk_cap(
        self,
        symbol: str,
        account_summary: Dict[str, Any],
        *,
        multiplier: Optional[float] = None,
    ) -> None:
        position = self._positions.get(symbol)
        if not position:
            return

        if multiplier is not None:
            try:
                numeric_multiplier = float(multiplier)
            except (TypeError, ValueError):
                numeric_multiplier = None
            if numeric_multiplier is not None and numeric_multiplier > 0:
                position.multiplier = numeric_multiplier

        try:
            amount = abs(float(position.amount))
        except (TypeError, ValueError):
            amount = 0.0
        if amount <= 0:
            position.reached_risk_cap = False
            return

        risk = self._estimate_position_risk(position)
        risk_unit = ensure_float(account_summary.get("risk_unit_R"), 0.0)
        max_risk_total = ensure_float(account_summary.get("max_risk_total_R"), 0.0)

        if risk == float("inf"):
            position.reached_risk_cap = True
            return

        if risk_unit > 0:
            if max_risk_total > 0:
                threshold = max_risk_total - risk_unit
                if threshold <= 0:
                    position.reached_risk_cap = True
                    return
                position.reached_risk_cap = not (risk < threshold)
            else:
                position.reached_risk_cap = False
        else:
            position.reached_risk_cap = False

    @staticmethod
    def _estimate_position_risk(position: Position) -> float:
        """Estimate risk without importing filter module (avoids circular dependency)."""

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

    def record_execution(
        self,
        symbol: str,
        decision: Dict[str, Any],
        order_result: Dict[str, Any],
    ) -> None:
        order_payload = order_result or {}
        position = self._positions.get(symbol)
        if not position:
            return
        action = str(decision.get("action") or decision.get("act") or "").upper()

        orders_payload = order_payload.get("sub_orders")
        if not isinstance(orders_payload, list):
            candidate = order_payload.get("orders")
            if isinstance(candidate, list):
                orders_payload = candidate
            elif candidate:
                orders_payload = [candidate]
            else:
                candidate = order_payload.get("fills")
                if isinstance(candidate, list):
                    orders_payload = candidate
        if not isinstance(orders_payload, list):
            single_order = order_payload.get("order")
            if isinstance(single_order, list):
                orders_payload = list(single_order)
            elif single_order:
                orders_payload = [single_order]
            else:
                orders_payload = []
        orders_payload = [item for item in orders_payload if item is not None]

        entry = {
            "timestamp": format_ts(utc_now().timestamp()),
            "decision": self._json_safe(decision),
            "orders": self._json_safe(orders_payload or []),
            "summary": self._json_safe(
                {
                    k: v
                    for k, v in {
                        "action": action,
                        "reason": decision.get("reason") or decision.get("rsn"),
                        "side": order_payload.get("side") or decision.get("side"),
                        "requested_qty": self._coerce_number(order_payload, [
                            "requested_qty",
                            "amount",
                            "qty",
                            "size",
                        ]),
                        "filled_qty": self._coerce_number(order_payload, [
                            "filled_qty",
                            "filled",
                            "executedQty",
                            "executed_quantity",
                            "qty",
                        ]),
                        "avg_price": self._coerce_number(order_payload, [
                            "avg_price",
                            "average",
                            "average_price",
                            "price",
                        ]),
                    }.items()
                    if v is not None
                }
            ),
        }
        position.execution_history.append(entry)
        self._apply_fill_price(position, action, order_payload, orders_payload)
        self._apply_decision_metadata(position, decision)

        if action == "ADD":
            position.triggered_targets = []
            base_state = {}
            if isinstance(position.trail_state, dict):
                base_risk = position.trail_state.get("base_risk")
                origin_sl = position.trail_state.get("origin_sl")
                if base_risk:
                    base_state["base_risk"] = base_risk
                if origin_sl:
                    base_state["origin_sl"] = origin_sl
            position.trail_state = base_state
        meta = decision.get("meta")
        if isinstance(meta, dict):
            target_value = meta.get("target")
            if target_value is not None:
                try:
                    numeric = float(target_value)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None and numeric not in position.triggered_targets:
                    position.triggered_targets.append(numeric)
            if meta.get("trail_activate"):
                trail_state = position.trail_state or {}
                trail_state["active"] = True
                trail_state["activated_at"] = format_ts(utc_now().timestamp())
                position.trail_state = trail_state
            if meta.get("be_activate"):
                trail_state = position.trail_state or {}
                trail_state["be_active"] = True
                trail_state["be_activated_at"] = format_ts(utc_now().timestamp())
                if "base_risk" not in trail_state:
                    trail_state["base_risk"] = abs(ensure_float(position.entry, 0.0) - ensure_float(position.sl, 0.0))
                position.trail_state = trail_state
            if meta.get("hybrid_trail_ready"):
                trail_state = position.trail_state or {}
                trail_state["hybrid_ready"] = True
                remaining_fraction = meta.get("hybrid_remaining_fraction")
                if remaining_fraction is not None:
                    trail_state["hybrid_remaining_fraction"] = remaining_fraction
                position.trail_state = trail_state
        position.revision += 1

    def _apply_decision_metadata(self, position: Position, decision: Optional[Dict[str, Any]]) -> None:
        if not position or not decision:
            return
        action = str(decision.get("action") or decision.get("act") or "").upper()

        actual_rr_value = decision.get("actual_rr")
        if actual_rr_value is not None:
            try:
                position.actual_rr = float(actual_rr_value)
            except (TypeError, ValueError):
                pass

        playbook = decision.get("playbook") or decision.get("pb")
        if playbook:
            position.playbook = playbook
        confidence = decision.get("confidence")
        if confidence is None:
            confidence = decision.get("conf")
        if confidence is not None:
            try:
                position.confidence = float(confidence)
            except (TypeError, ValueError):
                pass
        rr_value = decision.get("rr")
        if rr_value is not None:
            try:
                position.rr = float(rr_value)
            except (TypeError, ValueError):
                pass
        reason = decision.get("reason") or decision.get("rsn")
        if reason and not position.reason:
            position.reason = reason
        sl_value = decision.get("sl")
        if sl_value is not None:
            try:
                position.sl = float(sl_value)
            except (TypeError, ValueError):
                pass
        targets = decision.get("targets") or decision.get("tgts")
        if isinstance(targets, list):
            coerced_targets = []
            for value in targets:
                try:
                    coerced_targets.append(float(value))
                except (TypeError, ValueError):
                    continue
            if coerced_targets:
                position.targets = coerced_targets
                position.triggered_targets = [t for t in position.triggered_targets if t in position.targets]
        tp_policy = decision.get("tp_policy") or decision.get("tp_pol")
        if tp_policy:
            position.tp_policy = tp_policy
        tp_lock = decision.get("tp_lock")
        if isinstance(tp_lock, dict):
            position.tp_lock = dict(tp_lock)
        trailing = decision.get("trailing") or decision.get("trail")
        if action == "TRAIL_OFF":
            position.trailing = {}
            position.trail_state = {}
        elif isinstance(trailing, dict):
            position.trailing = dict(trailing)
        self._update_trailing_base_metrics(position, action)

    def _update_trailing_base_metrics(self, position: Position, action: str) -> None:
        trail_state = position.trail_state if isinstance(position.trail_state, dict) else {}
        if action == "TRAIL_OFF":
            return
        trailing_conf = position.trailing if isinstance(position.trailing, dict) else {}
        if not trailing_conf:
            return
        entry_value = ensure_float(getattr(position, "entry", None), None)
        sl_value = ensure_float(getattr(position, "sl", None), None)
        if entry_value is None or sl_value is None:
            return
        base_risk = abs(entry_value - sl_value)
        if base_risk <= 0:
            return
        if "base_risk" not in trail_state:
            trail_state["base_risk"] = base_risk
        if "origin_sl" not in trail_state:
            trail_state["origin_sl"] = sl_value
        position.trail_state = trail_state

    def remove(self, symbol: str) -> None:
        position = self._positions.pop(symbol, None)
        if not position:
            return
        try:
            self._last_exits[symbol] = utc_now().timestamp()
        except Exception:
            self._last_exits[symbol] = 0.0
        try:
            self._state_store.archive_position(symbol, position.to_dict())
        except Exception:
            self._logger.exception("Failed to archive position %s", symbol)

    def save(self) -> None:
        payload = {}
        for symbol, pos in list(self._positions.items()):
            if pos is None:
                self._positions.pop(symbol, None)
                continue
            payload[symbol] = pos.to_dict()
        self._state_store.save_positions(payload)

    def last_exit(self, symbol: str) -> Optional[float]:
        return self._last_exits.get(symbol)


__all__ = ["PositionManager"]
