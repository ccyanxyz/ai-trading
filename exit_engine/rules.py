"""Built-in exit rules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, Tuple
import math

from ..core.position import Position
from ..exit_engine.models import ExitContext, ExitDecision
from ..utils import ensure_float


class ExitRule(Protocol):
    """Interface implemented by exit rules."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        ...


def _position_size(position: Position) -> float:
    try:
        return abs(float(position.amount))
    except (TypeError, ValueError):
        return 0.0


class StopLossExitRule:
    """Fire when price crosses the AI-provided stop loss."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        sl = ensure_float(ctx.position.sl, None)
        if sl is None or sl <= 0:
            return None
        size = _position_size(ctx.position)
        if size <= 0:
            return None

        side = (ctx.position.side or "long").lower()
        price = ctx.price
        triggered = False
        if side == "long" and price <= sl:
            triggered = True
        elif side == "short" and price >= sl:
            triggered = True

        if not triggered:
            return None

        return ExitDecision(
            symbol=ctx.symbol,
            action="CLOSE",
            amount=size,
            reason="auto-triggered rule-based stop-loss",
            rule=self.__class__.__name__,
            metadata={"sl": sl, "price": price},
        )


class TargetExitRule:
    """Handles partial take-profits when price hits targets."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        targets = ctx.position.targets or []
        if not targets:
            return None

        size = _position_size(ctx.position)
        if size <= 0:
            return None

        policy = (ctx.position.tp_policy or "fixed").lower()
        triggered = ctx.position.triggered_targets or []
        remaining_targets = self._remaining_targets(targets, triggered)
        if not remaining_targets:
            return None

        side = (ctx.position.side or "long").lower()
        price = ctx.price

        if policy in {"hybrid", "trailing"}:
            decision = self._evaluate_hybrid(ctx, side, price, size, targets, triggered, remaining_targets)
        else:
            decision = self._evaluate_fixed(ctx, side, price, size, remaining_targets)

        return decision

    def _evaluate_fixed(
        self,
        ctx: ExitContext,
        side: str,
        price: float,
        size: float,
        remaining_targets: list[float],
    ) -> Optional[ExitDecision]:
        next_target = self._first_reached_target(side, price, remaining_targets)
        if next_target is None:
            return None

        fraction = round(1.0 / max(2, len(remaining_targets)), 2)
        amount = max(size * fraction, 0.0)
        if amount <= 0:
            return None

        return ExitDecision(
            symbol=ctx.symbol,
            action="PARTIAL_TP",
            amount=amount,
            reason="auto-triggered rule-based take-profit",
            rule=self.__class__.__name__,
            metadata={
                "target": next_target,
                "price": price,
                "fraction": fraction,
            },
        )

    def _evaluate_hybrid(
        self,
        ctx: ExitContext,
        side: str,
        price: float,
        size: float,
        all_targets: list[float],
        triggered_targets: list[float],
        remaining_targets: list[float],
    ) -> Optional[ExitDecision]:
        # policy = (ctx.position.tp_policy or "fixed").lower()
        # if policy == "trailing":
        #     return None

        sorted_targets = sorted(all_targets)
        remaining_sorted = sorted(remaining_targets)
        if not remaining_sorted:
            return None

        tolerance = 1e-8

        def _is_triggered(value: float) -> bool:
            return any(math.isclose(value, t, rel_tol=1e-6, abs_tol=tolerance) for t in triggered_targets)

        next_target = self._first_reached_target(side, price, remaining_sorted)
        if next_target is None:
            return None

        fraction = 0.0
        triggered_fraction = 0.0
        hybrid_ready = False

        if len(sorted_targets) == 1:
            already_triggered = _is_triggered(sorted_targets[0])
            if already_triggered:
                return None
            fraction = 0.5
            triggered_fraction = fraction
            hybrid_ready = True
        else:
            primary_targets = sorted_targets[:2]
            triggered_primary = sum(1 for t in primary_targets if _is_triggered(t))
            if triggered_primary >= 2:
                return None
            if triggered_primary == 0:
                selected_target = primary_targets[0]
            else:
                selected_target = primary_targets[1]
            if not math.isclose(selected_target, next_target, rel_tol=1e-6, abs_tol=tolerance):
                return None
            fraction = round(1.0 / 3.0, 2)
            triggered_fraction = fraction * (triggered_primary + 1)
            hybrid_ready = triggered_primary + 1 >= 2

        amount = max(size * fraction, 0.0)
        if amount <= 0:
            return None

        remaining_fraction = round(max(0.0, 1.0 - triggered_fraction), 2)

        return ExitDecision(
            symbol=ctx.symbol,
            action="PARTIAL_TP",
            amount=amount,
            reason="auto-triggered rule-based hybrid take-profit",
            rule=self.__class__.__name__,
            metadata={
                "target": next_target,
                "price": price,
                "fraction": fraction,
                "hybrid_trail_ready": hybrid_ready,
                "hybrid_remaining_fraction": remaining_fraction,
            },
        )

    @staticmethod
    def _remaining_targets(all_targets: list[float], triggered: list[float]) -> list[float]:
        remaining: list[float] = []
        for target in all_targets:
            if not any(math.isclose(target, trig, rel_tol=1e-6, abs_tol=1e-8) for trig in triggered):
                remaining.append(target)
        return remaining

    @staticmethod
    def _first_reached_target(side: str, price: float, targets: list[float]) -> Optional[float]:
        for target in sorted(targets):
            if side == "long" and price >= target:
                return target
            if side == "short" and price <= target:
                return target
        return None


class BreakEvenStopRule:
    """Push stop loss to break-even once price moves a configured R multiple."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        tp_lock = ctx.position.tp_lock or {}
        sl_to_be = ensure_float(tp_lock.get("sl_to_be"), None)
        if sl_to_be is None or sl_to_be <= 0:
            return None

        size = _position_size(ctx.position)
        if size <= 0:
            return None

        entry = ensure_float(ctx.position.entry, 0.0)
        sl = ensure_float(ctx.position.sl, 0.0)
        if entry <= 0 or sl <= 0:
            return None

        side = (ctx.position.side or "long").lower()
        price = ctx.price
        risk = entry - sl if side == "long" else sl - entry
        if risk <= 0:
            return None

        trail_state = ctx.position.trail_state or {}
        if bool(trail_state.get("be_active")):
            return None

        move = price - entry if side == "long" else entry - price
        if move < sl_to_be * risk:
            return None

        tolerance_basis = max(abs(price), abs(sl), 1.0)
        tolerance = tolerance_basis * 1e-6

        if side == "long":
            target_sl = min(entry, price - tolerance)
            if target_sl <= sl + tolerance:
                return None
        else:
            target_sl = max(entry, price + tolerance)
            if target_sl >= sl - tolerance:
                return None

        return ExitDecision(
            symbol=ctx.symbol,
            action="ADJUST_SL",
            sl=target_sl,
            reason="auto break-even stop",
            rule=self.__class__.__name__,
            metadata={
                "be_activate": True,
                "price": price,
                "move_distance": move,
                "sl_to_be": sl_to_be,
            },
        )


class TrailingExitRule:
    """Enables or adjusts trailing stops based on AI-provided parameters."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        trailing = ctx.position.trailing or {}
        tp_lock = ctx.position.tp_lock or {}
        if not trailing:
            return None

        size = _position_size(ctx.position)
        if size <= 0:
            return None

        entry = ensure_float(ctx.position.entry, 0.0)
        sl = ensure_float(ctx.position.sl, 0.0)
        if entry <= 0 or sl <= 0:
            return None

        side = (ctx.position.side or "long").lower()
        price = ctx.price
        self._touch_extrema(ctx.position, price)
        risk = entry - sl if side == "long" else sl - entry
        trail_state = ctx.position.trail_state or {}
        base_risk = ensure_float(trail_state.get("base_risk"), None)
        if (risk is None or risk <= 0) and base_risk and base_risk > 0:
            risk = base_risk
        if risk is None or risk <= 0:
            return None

        active = bool(trail_state.get("active"))
        hybrid_ready = bool(trail_state.get("hybrid_ready"))

        should_activate = False
        trail_after = ensure_float(tp_lock.get("trail_after"), None)
        if hybrid_ready and not active:
            should_activate = True
        elif not active:
            if trail_after is None:
                should_activate = True
            else:
                move = price - entry if side == "long" else entry - price
                if move >= trail_after * risk:
                    should_activate = True

        if not active and not should_activate:
            return None

        new_sl = self._compute_trailing_sl(ctx, trailing, side, price, sl)
        if new_sl is None:
            return None

        tolerance_basis = max(abs(price), abs(sl), 1.0)
        tolerance = tolerance_basis * 1e-6
        should_update = False
        if side == "long":
            if new_sl > sl + tolerance:
                should_update = True
        else:
            if new_sl < sl - tolerance:
                should_update = True
        if not should_update:
            return None

        move_delta = new_sl - sl
        action_metadata = {
            "price": price,
            "old_sl": sl,
            "new_sl": new_sl,
            "move_distance": abs(move_delta),
        }
        if should_activate and not active:
            action_metadata["trail_activate"] = True
            if hybrid_ready:
                action_metadata["hybrid"] = True
        else:
            action_metadata["trail_adjust"] = True

        trail_state_snapshot = ctx.position.trail_state if isinstance(ctx.position.trail_state, dict) else {}
        if isinstance(trail_state_snapshot, dict):
            last_atr = ensure_float(trail_state_snapshot.get("last_atr"), None)
            if last_atr is not None:
                action_metadata["atr_value"] = last_atr
                atr_source = trail_state_snapshot.get("last_atr_source")
                if atr_source:
                    action_metadata["atr_source"] = atr_source

        delta_text = f"{move_delta:+.4f}"
        reason = f"auto-trailing: {sl:.4f} -> {new_sl:.4f} ({delta_text})"

        return ExitDecision(
            symbol=ctx.symbol,
            action="ADJUST_SL",
            sl=new_sl,
            reason=reason,
            rule=self.__class__.__name__,
            metadata=action_metadata,
        )

    def _compute_trailing_sl(
        self,
        ctx: ExitContext,
        trailing: Dict[str, Any],
        side: str,
        price: float,
        current_sl: float,
    ) -> Optional[float]:
        kind = str(trailing.get("kind") or "atr").lower()
        anchor = str(trailing.get("anc") or trailing.get("anchor") or "close").lower()
        payload = ctx.payload or {}
        indicators = payload.get("indicators") or {}

        anchor_price = self._anchor_price(
            anchor,
            side,
            price,
            ctx.position,
            payload,
            indicators,
            trailing,
        )
        if anchor_price is None:
            anchor_price = price

        if kind == "atr":
            atr_lookup = self._atr_value(trailing, indicators)
            multi = ensure_float(trailing.get("multi"), None)
            if atr_lookup is None or multi is None:
                return None
            atr_value, atr_source = atr_lookup
            self._record_trailing_atr(ctx.position, atr_value, atr_source)
            adjustment = atr_value * multi
            return anchor_price - adjustment if side == "long" else anchor_price + adjustment

        if kind == "pct":
            dist = ensure_float(trailing.get("dist"), None)
            if dist is None or dist <= 0:
                return None
            if side == "long":
                return price * (1 - dist)
            return price * (1 + dist)

        if kind == "chandelier":
            atr_lookup = self._atr_value(trailing, indicators)
            multi = ensure_float(trailing.get("multi"), None)
            window = int(trailing.get("window", 22))
            ref = self._high_low_reference(side, ctx.position, window)
            if atr_lookup is None or multi is None or ref is None:
                return None
            atr_value, atr_source = atr_lookup
            self._record_trailing_atr(ctx.position, atr_value, atr_source)
            return ref - atr_value * multi if side == "long" else ref + atr_value * multi

        return None

    def _atr_value(
        self,
        trailing: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> Optional[Tuple[float, Optional[str]]]:
        atr = indicators.get("atr") if isinstance(indicators, dict) else None
        if not isinstance(atr, dict):
            return None
        priority = ["1h", "4h", "15m", "day"]
        for key in priority:
            if key in atr:
                value = ensure_float(atr.get(key), None)
                if value is not None and value > 0:
                    return value, key
        for key, value in atr.items():
            numeric = ensure_float(value, None)
            if numeric is not None and numeric > 0:
                return numeric, str(key)
        return None

    def _anchor_price(
        self,
        anchor: str,
        side: str,
        price: float,
        position: Position,
        payload: Dict[str, Any],
        indicators: Dict[str, Any],
        trailing: Dict[str, Any],
    ) -> Optional[float]:
        if anchor == "close":
            return price
        if anchor == "ema20":
            ema = indicators.get("ema") if isinstance(indicators, dict) else None
            if isinstance(ema, dict):
                # prefer 1h 20-period if available
                preferred_keys = ["1h_20", "1h20", "4h_20", "4h20"]
                for key in preferred_keys:
                    series = ema.get(key)
                    if isinstance(series, Sequence) and series:
                        return ensure_float(series[-1], None)
                for series in ema.values():
                    if isinstance(series, Sequence) and series:
                        value = ensure_float(series[-1], None)
                        if value is not None:
                            return value
        if anchor == "high_low":
            return self._high_low_reference(
                side,
                position,
                int(trailing.get("window", 22)),
            )
        return price

    def _high_low_reference(
        self,
        side: str,
        position: Position,
        window: int,
    ) -> Optional[float]:
        high, low = self._position_extrema(position)
        if side == "long":
            return high
        return low

    @staticmethod
    def _position_extrema(position: Position) -> Tuple[Optional[float], Optional[float]]:
        trail_state = position.trail_state if isinstance(position.trail_state, dict) else {}
        extrema = trail_state.get("price_extrema") if isinstance(trail_state.get("price_extrema"), dict) else {}
        high_val = ensure_float(extrema.get("high"), None)
        low_val = ensure_float(extrema.get("low"), None)
        return high_val, low_val

    @staticmethod
    def _touch_extrema(position: Position, price: float) -> None:
        try:
            numeric = float(price)
        except (TypeError, ValueError):
            return
        trail_state = position.trail_state if isinstance(position.trail_state, dict) else {}
        if not isinstance(trail_state, dict):
            trail_state = {}
        extrema = trail_state.get("price_extrema") if isinstance(trail_state.get("price_extrema"), dict) else {}
        high_val = ensure_float(extrema.get("high"), None)
        if high_val is None or numeric > high_val:
            extrema["high"] = numeric
        low_val = ensure_float(extrema.get("low"), None)
        if low_val is None or numeric < low_val:
            extrema["low"] = numeric
        trail_state["price_extrema"] = extrema
        position.trail_state = trail_state

    @staticmethod
    def _record_trailing_atr(position: Position, value: float, source: Optional[str]) -> None:
        numeric = ensure_float(value, None)
        if numeric is None or numeric <= 0:
            return
        trail_state = position.trail_state if isinstance(position.trail_state, dict) else {}
        if not isinstance(trail_state, dict):
            trail_state = {}
        trail_state["last_atr"] = numeric
        if source:
            trail_state["last_atr_source"] = source
        position.trail_state = trail_state
