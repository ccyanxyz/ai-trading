"""Decision execution pipeline for AI strategies."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from ..utils import ensure_float, format_ts, normalize_decision_symbol, utc_now
from ..filters.pre_filters import PreExecutionFilter
from ..core.interfaces import SymbolData
from ..core.position import Position
from ..core.context import StrategyContext
from ..exit_engine import ExitContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .strategy import AITradingStrategy


class DecisionPipeline:
    """Coordinates a single AI evaluation cycle for a strategy."""

    def __init__(self, strategy: "AITradingStrategy") -> None:
        self._strategy = strategy
        self._logger = strategy.logger

    async def run(self, context: StrategyContext, *, force: bool = False) -> StrategyContext:
        strategy = self._strategy
        manager = context.position_manager or getattr(strategy, "_position_manager", None)
        pre_filter = getattr(strategy, "_pre_filter", None)
        pre_filter_enabled = bool(getattr(strategy, "_pre_filter_enabled", False))
        execution_guard = getattr(strategy, "_execution_guard", None) if getattr(strategy, "_execution_guard_enabled", False) else None
        exit_engine = getattr(strategy, "_exit_engine", None) if getattr(strategy, "_exit_engine_enabled", False) else None
        if not strategy._ai_client:
            strategy.logger.debug("AI client未初始化，跳过执行")
            return context

        (
            account_snapshot,
            account_summary,
            exchange_positions,
        ) = await self._load_account_state(context, manager)

        (
            symbol_data,
            pre_auto_decisions,
            pre_auto_orders,
        ) = await self._collect_symbol_data(
            context,
            account_snapshot,
            account_summary,
            exchange_positions,
            manager,
            pre_filter,
            pre_filter_enabled,
            exit_engine,
            execution_guard,
            force,
        )

        self._persist_pre_auto_decisions(pre_auto_decisions, pre_auto_orders, account_snapshot)

        if not symbol_data:
            strategy.logger.debug("No symbols require AI query this cycle (force=%s)", force)
            context.latest_decisions = list(pre_auto_decisions)
            return context

        batches, batch_payloads = self._prepare_batches(symbol_data, account_snapshot)
        if not batches:
            strategy.logger.warning("Batch planner returned no batches; skipping run")
            return context

        (
            executed,
            processed_updated,
            all_decisions,
            response_ids,
            response_timestamps,
        ) = await self._invoke_ai_and_execute(
            batches,
            batch_payloads,
            context,
            account_snapshot,
            account_summary,
            exchange_positions,
            execution_guard,
            pre_auto_orders,
            pre_auto_decisions,
        )

        self._finalize_cycle(
            context,
            account_snapshot,
            all_decisions,
            response_ids,
            response_timestamps,
            batch_payloads,
            processed_updated,
            executed,
        )
        return context

    async def _load_account_state(
        self,
        context: StrategyContext,
        manager: Optional[Any],
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        strategy = self._strategy
        await strategy._ensure_adapter()
        raw_account_state = await strategy.adapter.fetch_account_state()
        account_snapshot = strategy._account_formatter.format(raw_account_state)
        strategy._apply_risk_budget(account_snapshot)
        strategy._last_account_snapshot = account_snapshot
        context.account_snapshot = account_snapshot
        account_summary = account_snapshot.summary or {}
        raw_positions = account_snapshot.raw.get("positions") if account_snapshot.raw else None
        exchange_positions: Dict[str, Any] = {}
        if isinstance(raw_positions, list):
            for position_payload in raw_positions:
                if not isinstance(position_payload, dict):
                    continue
                symbol_text = position_payload.get("symbol") or position_payload.get("sym")
                if not symbol_text:
                    continue
                normalized_symbol = strategy.normalize_symbol(str(symbol_text))
                exchange_positions[normalized_symbol] = position_payload
        if manager:
            for sym, payload in exchange_positions.items():
                updated = manager.upsert(sym, payload, update_revision=False)
                if updated is None:
                    manager.remove(sym)
                    context.positions.pop(sym, None)
                    continue
                manager.update_risk_cap(
                    sym,
                    account_summary,
                    multiplier=ensure_float(payload.get("multiplier"), None)
                    if isinstance(payload, dict)
                    else None,
                )
                try:
                    remaining = abs(float(updated.amount))
                except (TypeError, ValueError):
                    remaining = 0.0
                if remaining <= 0:
                    manager.remove(sym)
                    context.positions.pop(sym, None)
                    continue
                if sym not in context.symbols:
                    tracked = strategy._ensure_symbol_tracking(sym)
                    if tracked not in context.symbols:
                        context.symbols.append(tracked)
                        context.symbols.sort()
                    context.last_processed.setdefault(tracked, 0)
                context.positions[sym] = updated
            context.positions = manager.positions
        return account_snapshot, account_summary, exchange_positions

    async def _collect_symbol_data(
        self,
        context: StrategyContext,
        account_snapshot: Any,
        account_summary: Dict[str, Any],
        exchange_positions: Dict[str, Any],
        manager: Optional[Any],
        pre_filter: Optional[PreExecutionFilter],
        pre_filter_enabled: bool,
        exit_engine: Optional[Any],
        execution_guard: Optional[Any],
        force: bool,
    ) -> Tuple[List[SymbolData], List[Dict[str, Any]], List[Dict[str, Any]]]:
        strategy = self._strategy
        history_conf = strategy.config.get("history") or {}
        fetch_concurrency_raw = history_conf.get("fetch_concurrency", 5)
        try:
            fetch_concurrency = int(fetch_concurrency_raw)
        except (TypeError, ValueError):
            fetch_concurrency = 5
        if fetch_concurrency <= 0:
            fetch_concurrency = 1
        semaphore = asyncio.Semaphore(fetch_concurrency)

        async def collect_symbol(sym: str) -> Tuple[str, Optional[SymbolData], Optional[Any]]:
            position_local = context.positions.get(sym)
            if hasattr(strategy.adapter, "is_market_open"):
                try:
                    market_open = await strategy.adapter.is_market_open(sym)
                except Exception:
                    strategy.logger.exception("Market hours check failed for %s", sym)
                    market_open = True
                if not market_open:
                    strategy.logger.debug("Skip symbol %s: market closed", sym)
                    return sym, None, position_local
            try:
                async with semaphore:
                    data_local = await strategy.adapter.collect_symbol_data(
                        sym,
                        account_snapshot.raw,
                        position=position_local,
                        risk=strategy._risk_conf,
                        constraints=strategy._constraints,
                    )
            except Exception:
                strategy.logger.exception("Failed to collect symbol data for %s", sym)
                return sym, None, position_local
            return sym, data_local, position_local

        tasks: List[asyncio.Task[Tuple[str, Optional[SymbolData], Optional[Any]]]] = [
            asyncio.create_task(collect_symbol(sym)) for sym in context.symbols
        ] if context.symbols else []

        symbol_results: List[Tuple[str, Optional[SymbolData], Optional[Any]]] = []
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(gathered):
                sym = context.symbols[idx]
                if isinstance(result, Exception):
                    strategy.logger.exception("Symbol data task failed for %s", sym, exc_info=result)
                    continue
                symbol_results.append(result)

        symbol_data: List[SymbolData] = []
        pre_auto_decisions: List[Dict[str, Any]] = []
        pre_auto_orders: List[Dict[str, Any]] = []

        for sym, data, original_position in symbol_results:
            position = context.positions.get(sym) or original_position
            if data is None:
                continue
            data.position = position
            if manager:
                meta_multiplier = None
                if isinstance(data.meta, dict):
                    meta_multiplier = ensure_float(data.meta.get("multiplier"), None)
                manager.update_risk_cap(
                    sym,
                    account_summary,
                    multiplier=meta_multiplier,
                )
            if manager:
                last_exit_ts = manager.last_exit(sym)
                if last_exit_ts is not None:
                    data.meta.setdefault("last_exit_ts", last_exit_ts)
            allowed_actions: Optional[Set[str]] = None
            if pre_filter:
                if pre_filter_enabled:
                    skip, reason = pre_filter.should_skip_ai(sym, data, account_snapshot, position)
                    if skip:
                        strategy.logger.debug("Skip symbol %s due to pre-filter: %s", sym, reason)
                        continue
                allowed_actions = set(
                    pre_filter.get_allowed_actions(
                        sym,
                        position,
                        account_snapshot.summary or {},
                    )
                )
            if not allowed_actions:
                allowed_actions = self._fallback_allowed_actions(position)
            allowed_actions = {str(item).upper() for item in allowed_actions if item}
            if not allowed_actions:
                allowed_actions = {"HOLD"}
            allowed_actions_list = sorted(allowed_actions)
            data.meta["allowed_actions"] = allowed_actions_list

            exit_executed = False
            if (
                exit_engine
                and manager
                and position
                and getattr(strategy, "_exit_rules_pipeline_enabled", True)
            ):
                exit_executed = await self._apply_exit_rules(
                    strategy,
                    exit_engine,
                    data,
                    position,
                    allowed_actions_list,
                    manager,
                    execution_guard,
                    account_snapshot,
                    exchange_positions.get(sym),
                    pre_auto_decisions,
                    pre_auto_orders,
                    context,
                )
            if exit_executed:
                position = context.positions.get(sym)
                if not position:
                    continue
                    try:
                        remaining_amount = abs(float(position.amount))
                    except (TypeError, ValueError):
                        remaining_amount = 0.0
                    if remaining_amount <= 0:
                        continue
            last_processed_ts = context.last_processed.get(sym, 0)
            if not strategy._schedule_policy.should_process(
                strategy,
                sym,
                data,
                last_processed_ts,
                force=force,
            ):
                strategy.logger.debug(
                    "Skip symbol %s: schedule deferred (force=%s, last_ts=%s, last_processed=%s)",
                    sym,
                    force,
                    data.last_ts,
                    last_processed_ts,
                )
                continue
            symbol_data.append(data)

        return symbol_data, pre_auto_decisions, pre_auto_orders

    def _persist_pre_auto_decisions(
        self,
        pre_auto_decisions: Sequence[Dict[str, Any]],
        pre_auto_orders: Sequence[Dict[str, Any]],
        account_snapshot: Any,
    ) -> None:
        if not pre_auto_decisions:
            return
        strategy = self._strategy
        for exit_decision in pre_auto_decisions:
            strategy.logger.info(
                "Auto exit decision queued: sym=%s action=%s reason=%s",
                exit_decision.get("symbol") or exit_decision.get("sym"),
                exit_decision.get("action"),
                exit_decision.get("reason"),
            )
        timestamp_iso = format_ts(utc_now().timestamp())
        account_payload = strategy._json_safe(account_snapshot.raw or account_snapshot.summary or {})
        strategy.state_store.persist_decision(
            timestamp_iso,
            ["auto_exit"],
            account_payload,
            list(pre_auto_decisions),
        )
        if pre_auto_orders:
            strategy.state_store.append_execution_log(
                {"ts": timestamp_iso, "orders": list(pre_auto_orders)}
            )

    def _prepare_batches(
        self, symbol_data: Sequence[SymbolData], account_snapshot: Any
    ) -> Tuple[List[Sequence[SymbolData]], List[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]]]]:
        strategy = self._strategy
        if strategy._batch_planner:
            batches = list(strategy._batch_planner.plan_batches(strategy, symbol_data))
        else:
            ai_conf = strategy.config.get("ai") or {}
            batch_size = int(ai_conf.get("batch_size", 0))
            if batch_size <= 0:
                batch_size = len(symbol_data)
            batches = [
                symbol_data[i : i + batch_size]
                for i in range(0, len(symbol_data), batch_size)
            ]
        if not batches:
            return [], []
        strategy.logger.info("Dispatching AI in %d batch(es)", len(batches))
        batch_payloads: List[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]]] = []
        for idx, batch in enumerate(batches):
            timestamp_iso = format_ts(utc_now().timestamp())
            if strategy._payload_builder:
                payload = strategy._payload_builder.build_payload(
                    strategy,
                    timestamp_iso,
                    account_snapshot,
                    batch,
                )
            else:
                payload = {
                    "timestamp": timestamp_iso,
                    "account": account_snapshot.prompt_payload,
                    "pairs": [item.payload for item in batch],
                }
            if strategy._envelope_adapter:
                encoded_payload = strategy._envelope_adapter.encode_request(
                    strategy,
                    payload,
                    account=account_snapshot,
                    symbols=batch,
                )
            else:
                encoded_payload = payload
            batch_payloads.append((idx, timestamp_iso, payload, encoded_payload, batch))
            strategy.logger.debug(
                "Prepared AI batch %d with symbols: %s",
                idx,
                ", ".join(item.symbol for item in batch),
            )
        return batches, batch_payloads

    async def _invoke_ai_and_execute(
        self,
        batches: Sequence[Sequence[SymbolData]],
        batch_payloads: Sequence[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]]],
        context: StrategyContext,
        account_snapshot: Any,
        account_summary: Dict[str, Any],
        exchange_positions: Dict[str, Any],
        execution_guard: Optional[Any],
        pre_auto_orders: Sequence[Dict[str, Any]],
        pre_auto_decisions: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]], List[str], List[str]]:
        executed: List[Dict[str, Any]] = list(pre_auto_orders)
        all_decisions: List[Dict[str, Any]] = list(pre_auto_decisions)
        response_ids: List[str] = []
        response_timestamps: List[str] = []
        processed_updated = False
        results = await self._invoke_ai_batches(batch_payloads)
        manager = context.position_manager or getattr(self._strategy, "_position_manager", None)

        for batch_meta, result in results:
            decisions, response_id = self._parse_batch_result(batch_meta, result, account_snapshot)
            if decisions is None:
                continue

            all_decisions.extend(decisions)
            if response_id:
                response_ids.append(str(response_id))
                response_timestamps.append(batch_meta[1])

            if await self._process_decisions_for_batch(
                batch_meta[4],
                decisions,
                context,
                account_snapshot,
                account_summary,
                exchange_positions,
                execution_guard,
                manager,
                executed,
            ):
                processed_updated = True

        return executed, processed_updated, all_decisions, response_ids, response_timestamps

    def _finalize_cycle(
        self,
        context: StrategyContext,
        account_snapshot: Any,
        all_decisions: Sequence[Dict[str, Any]],
        response_ids: Sequence[str],
        response_timestamps: Sequence[str],
        batch_payloads: Sequence[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]]],
        processed_updated: bool,
        executed: Sequence[Dict[str, Any]],
    ) -> None:
        strategy = self._strategy
        if response_ids:
            persist_timestamp = response_timestamps[0] if response_timestamps else batch_payloads[0][1]
            account_payload = strategy._json_safe(account_snapshot.raw or account_snapshot.summary or {})
            strategy.state_store.persist_decision(
                persist_timestamp,
                list(response_ids),
                account_payload,
                list(all_decisions),
            )
        if processed_updated:
            strategy.state_store.save_last_processed(context.last_processed)
        context.latest_decisions = list(all_decisions)
        run_time = utc_now()
        strategy._last_run_time = run_time
        context.last_run_time = run_time
        if executed:
            strategy.state_store.append_execution_log(
                {"ts": format_ts(run_time.timestamp()), "orders": list(executed)}
            )
        with contextlib.suppress(Exception):
            strategy._position_manager.save()

    async def _invoke_ai_batches(
        self,
        batch_payloads: Sequence[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]]],
    ) -> List[Tuple[Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]], Any]]:
        strategy = self._strategy
        results = await asyncio.gather(
            *(strategy._invoke_ai(encoded) for _, _, _, encoded, _ in batch_payloads),
            return_exceptions=True,
        )
        return list(zip(batch_payloads, results))

    def _parse_batch_result(
        self,
        batch_meta: Tuple[int, str, Dict[str, Any], Dict[str, Any], Sequence[SymbolData]],
        result: Any,
        account_snapshot: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        strategy = self._strategy
        idx, _, _source_payload, encoded_payload, batch = batch_meta
        if isinstance(result, Exception):
            strategy.logger.error("AI batch %d failed: %s", idx, result)
            return None, None
        response_id, decoded_payload = result
        decisions = strategy._envelope_adapter.decode_response(
            strategy,
            decoded_payload,
            payload=encoded_payload,
            account=account_snapshot,
            symbols=batch,
        )
        strategy.logger.info(
            "AI response received: batch=%d items=%d",
            idx,
            len(decisions) if isinstance(decisions, list) else 0,
        )
        if not isinstance(decisions, list):
            strategy.logger.warning(
                "AI batch %d returned non-list result: %s", idx, decisions
            )
            return None, response_id
        return decisions, response_id

    async def _process_decisions_for_batch(
        self,
        batch: Sequence[SymbolData],
        decisions: Sequence[Dict[str, Any]],
        context: StrategyContext,
        account_snapshot: Any,
        account_summary: Dict[str, Any],
        exchange_positions: Dict[str, Any],
        execution_guard: Optional[Any],
        manager: Optional[Any],
        executed: List[Dict[str, Any]],
    ) -> bool:
        processed_updated = False
        strategy = self._strategy
        for data in batch:
            decision = next(
                (d for d in decisions if normalize_decision_symbol(d) == data.symbol),
                None,
            )
            if decision is None:
                if context.last_processed.get(data.symbol) != data.last_ts:
                    context.last_processed[data.symbol] = data.last_ts
                    processed_updated = True
                strategy._schedule_policy.record_success(strategy, data.symbol, data, None)
                continue
            if await self._process_single_decision(
                data,
                decision,
                context,
                account_snapshot,
                account_summary,
                exchange_positions,
                execution_guard,
                manager,
                executed,
            ):
                processed_updated = True
        return processed_updated

    async def _process_single_decision(
        self,
        data: SymbolData,
        decision: Dict[str, Any],
        context: StrategyContext,
        account_snapshot: Any,
        account_summary: Dict[str, Any],
        exchange_positions: Dict[str, Any],
        execution_guard: Optional[Any],
        manager: Optional[Any],
        executed: List[Dict[str, Any]],
    ) -> bool:
        strategy = self._strategy
        sym = data.symbol
        position = context.positions.get(sym)
        action = str(decision.get("action") or decision.get("act") or "")
        action_upper = action.upper()
        reason = decision.get("reason") or decision.get("rsn")
        playbook = decision.get("playbook") or decision.get("pb")
        strategy.logger.info(
            "AI decision parsed: sym=%s act=%s pb=%s rsn=%s",
            sym,
            action,
            playbook or "",
            reason or "",
        )

        processed_updated = False
        self._ensure_close_quantity(action_upper, decision, position)

        is_entry_action = self._is_entry_action(action_upper, position)
        if is_entry_action:
            proceed, flag = self._validate_entry_rr(
                data,
                decision,
                action_upper,
                position,
                context,
            )
            processed_updated |= flag
            if not proceed:
                return processed_updated

        proceed, flag = self._handle_wait_or_none(action_upper, data, decision, context)
        processed_updated |= flag
        if not proceed:
            return processed_updated

        allowed_actions = self._resolve_allowed_actions(data.meta.get("allowed_actions"))
        proceed, flag = self._validate_with_execution_guard(
            sym,
            decision,
            account_snapshot,
            position,
            allowed_actions,
            execution_guard,
            context,
            data,
        )
        processed_updated |= flag
        if not proceed:
            return processed_updated

        proceed, flag = self._handle_add_action(
            action_upper,
            decision,
            position,
            data,
            account_summary,
            sym,
        )
        processed_updated |= flag
        if not proceed:
            return processed_updated

        provisional_position = None
        used_provisional = False
        if self._needs_provisional_position(decision, position):
            provisional_position = strategy.position_manager.upsert(
                sym,
                {
                    "symbol": sym,
                    "amount": decision.get("amount") or 0.0,
                    "entry": decision.get("price") or decision.get("entry"),
                    "side": decision.get("side") or decision.get("action"),
                },
                update_revision=False,
            )
            if provisional_position:
                used_provisional = True

        proceed, flag = await self._execute_order_for_decision(
            data,
            decision,
            action_upper,
            position,
            context,
            account_snapshot,
            manager,
            executed,
            used_provisional,
            sym,
        )
        processed_updated |= flag
        if not proceed:
            return processed_updated

        if context.last_processed.get(data.symbol) != data.last_ts:
            context.last_processed[data.symbol] = data.last_ts
            processed_updated = True
        strategy._schedule_policy.record_success(strategy, data.symbol, data, decision)
        return processed_updated

    def _ensure_close_quantity(
        self,
        action_upper: str,
        decision: Dict[str, Any],
        position: Optional[Position],
    ) -> None:
        if action_upper != "CLOSE" or not position:
            return
        if ensure_float(decision.get("amount"), 0.0) > 0:
            return
        close_amount = ensure_float(getattr(position, "amount", 0.0), 0.0)
        if close_amount and abs(close_amount) > 0:
            decision["amount"] = abs(close_amount)

    @staticmethod
    def _is_entry_action(action_upper: str, position: Optional[Position]) -> bool:
        if action_upper in {"BUY", "SELL"}:
            amount = ensure_float(getattr(position, "amount", 0.0), 0.0) if position else 0.0
            if position is None or abs(amount) <= 0:
                return True
        if action_upper == "ADD":
            return True
        return False

    def _validate_entry_rr(
        self,
        data: SymbolData,
        decision: Dict[str, Any],
        action_upper: str,
        position: Optional[Position],
        context: StrategyContext,
    ) -> Tuple[bool, bool]:
        strategy = self._strategy
        min_rr_required = ensure_float(strategy._constraints.get("min_rr"), 0.0)
        if min_rr_required <= 0:
            return True, False
        decision_rr = ensure_float(decision.get("rr"), None)
        price_candidate = ensure_float(decision.get("price") or decision.get("entry"), None)
        if price_candidate is None or price_candidate <= 0:
            price_candidate = ensure_float(data.last_price, None)
        sl_value = ensure_float(decision.get("sl"), None)
        targets_raw = decision.get("targets") or decision.get("tgts")
        targets: List[float] = []
        if isinstance(targets_raw, list):
            for tgt in targets_raw:
                try:
                    targets.append(float(tgt))
                except (TypeError, ValueError):
                    continue
        actual_rr: Optional[float] = None
        if (
            price_candidate is not None
            and price_candidate > 0
            and sl_value is not None
            and targets
        ):
            avg_target = sum(targets) / len(targets)
            if action_upper in {"BUY", "ADD"}:
                risk = price_candidate - sl_value
                reward = avg_target - price_candidate
            else:
                risk = sl_value - price_candidate
                reward = price_candidate - avg_target
            if risk > 0 and reward > 0:
                actual_rr = reward / risk
        if actual_rr is not None:
            decision["actual_rr"] = actual_rr
        rr_ok = decision_rr is not None and decision_rr >= min_rr_required
        actual_ok = actual_rr is not None and actual_rr >= min_rr_required
        strategy.logger.info(f"decision_rr: {decision_rr}, actual_rr: {actual_rr}")
        if actual_ok:
            return True, False
        strategy.logger.warning(
            "Skip %s: RR validation failed (decision_rr=%s actual_rr=%s min_rr=%.2f)",
            data.symbol,
            decision_rr,
            actual_rr,
            min_rr_required,
        )
        processed_updated = False
        if context.last_processed.get(data.symbol) != data.last_ts:
            context.last_processed[data.symbol] = data.last_ts
            processed_updated = True
        strategy._schedule_policy.record_success(strategy, data.symbol, data, decision)
        return False, processed_updated

    def _handle_wait_or_none(
        self,
        action_upper: str,
        data: SymbolData,
        decision: Dict[str, Any],
        context: StrategyContext,
    ) -> Tuple[bool, bool]:
        if action_upper not in {"WAIT", "NONE"}:
            return True, False
        processed_updated = False
        if context.last_processed.get(data.symbol) != data.last_ts:
            context.last_processed[data.symbol] = data.last_ts
            processed_updated = True
        self._strategy._schedule_policy.record_success(self._strategy, data.symbol, data, decision)
        return False, processed_updated

    @staticmethod
    def _resolve_allowed_actions(raw_actions: Optional[Iterable[Any]]) -> Set[str]:
        actions = {str(item).upper() for item in raw_actions or [] if item}
        return actions if actions else {"HOLD"}

    def _validate_with_execution_guard(
        self,
        sym: str,
        decision: Dict[str, Any],
        account_snapshot: Any,
        position: Optional[Position],
        allowed_actions: Set[str],
        execution_guard: Optional[Any],
        context: StrategyContext,
        data: SymbolData,
    ) -> Tuple[bool, bool]:
        if not execution_guard:
            return True, False
        valid, guard_reason = execution_guard.validate_decision(
            sym,
            decision,
            account_snapshot,
            position,
            allowed_actions,
        )
        if not valid:
            self._strategy.logger.warning(
                "Execution guard blocked decision for %s: %s",
                sym,
                guard_reason,
            )
            processed_updated = False
            if context.last_processed.get(data.symbol) != data.last_ts:
                context.last_processed[data.symbol] = data.last_ts
                processed_updated = True
            self._strategy._schedule_policy.record_success(self._strategy, data.symbol, data, decision)
            return False, processed_updated
        if decision.get("amount") is None and decision.get("action") and decision["action"].upper() in {"BUY", "SELL", "ADD", "REDUCE", "CLOSE", "PARTIAL_TP"}:
            sized_qty = execution_guard.calc_position_size(
                decision,
                account_snapshot,
                data,
                position,
            )
            if sized_qty <= 0:
                self._strategy.logger.warning(
                    "Execution guard could not derive size for %s action=%s",
                    sym,
                    decision.get("action"),
                )
                processed_updated = False
                if context.last_processed.get(data.symbol) != data.last_ts:
                    context.last_processed[data.symbol] = data.last_ts
                    processed_updated = True
                self._strategy._schedule_policy.record_success(self._strategy, data.symbol, data, decision)
                return False, processed_updated
            decision["amount"] = sized_qty
        return True, False

    def _handle_add_action(
        self,
        action_upper: str,
        decision: Dict[str, Any],
        position: Optional[Position],
        data: SymbolData,
        account_summary: Dict[str, Any],
        sym: str,
    ) -> Tuple[bool, bool]:
        if action_upper != "ADD":
            return True, False
        sl_candidate = ensure_float(decision.get("sl"), 0.0)
        if sl_candidate <= 0:
            self._strategy.logger.warning("Skip ADD for %s: invalid sl", sym)
            return False, False
        tp_policy = decision.get("tp_policy") or decision.get("tp_pol")
        if not isinstance(tp_policy, str) or not tp_policy:
            self._strategy.logger.warning("Skip ADD for %s: missing tp_policy", sym)
            return False, False
        targets = decision.get("targets") or decision.get("tgts")
        if not isinstance(targets, list) or not targets:
            self._strategy.logger.warning("Skip ADD for %s: missing targets", sym)
            return False, False
        decision["tp_policy"] = tp_policy
        decision["targets"] = targets
        current_amount = ensure_float(getattr(position, "amount", 0.0), 0.0) if position else 0.0
        reference_price = ensure_float(getattr(position, "entry", None), None) if position else None
        if reference_price is None or reference_price <= 0:
            reference_price = ensure_float(getattr(position, "mark", None), None) if position else None
        if reference_price is None or reference_price <= 0:
            reference_price = ensure_float(data.last_price, 0.0)
        if reference_price <= 0:
            self._strategy.logger.warning("Skip ADD for %s: invalid reference price", sym)
            return False, False
        multiplier = ensure_float(getattr(position, "multiplier", None), 1.0) if position else 1.0
        if multiplier <= 0:
            multiplier = 1.0
        risk_per_contract = abs(reference_price - sl_candidate) * multiplier
        if risk_per_contract <= 0:
            self._strategy.logger.warning("Skip ADD for %s: zero risk per contract", sym)
            return False, False
        risk_unit = ensure_float(account_summary.get("risk_unit_R"), 0.0)
        max_total = ensure_float(account_summary.get("max_risk_total_R"), 0.0)
        current_risk = PreExecutionFilter._estimate_position_risk(position) if position else 0.0
        if current_risk == float("inf"):
            self._strategy.logger.warning("Skip ADD for %s: current risk undefined", sym)
            return False, False
        available_total = float("inf")
        if max_total > 0:
            available_total = max(0.0, max_total - current_risk)
            if available_total <= 0:
                self._strategy.logger.info("Skip ADD for %s: no remaining risk budget", sym)
                return False, False
        target_risk = risk_unit if risk_unit > 0 else available_total
        if max_total > 0:
            target_risk = min(target_risk, available_total)
        if target_risk == float("inf"):
            target_risk = risk_per_contract
        if target_risk <= 0 or not (target_risk < float("inf")):
            self._strategy.logger.info("Skip ADD for %s: target risk not positive", sym)
            return False, False
        computed_amount = target_risk / risk_per_contract
        if current_amount < 0:
            computed_amount = -abs(computed_amount)
        else:
            computed_amount = abs(computed_amount)
        if computed_amount == 0:
            self._strategy.logger.warning("Skip ADD for %s: computed amount zero", sym)
            return False, False
        decision["amount"] = computed_amount
        return True, False

    @staticmethod
    def _needs_provisional_position(decision: Dict[str, Any], position: Optional[Position]) -> bool:
        return bool(decision.get("action")) and decision["action"].upper() in {"BUY", "SELL"} and not position

    async def _execute_order_for_decision(
        self,
        data: SymbolData,
        decision: Dict[str, Any],
        action_upper: str,
        position: Optional[Position],
        context: StrategyContext,
        account_snapshot: Any,
        manager: Optional[Any],
        executed: List[Dict[str, Any]],
        used_provisional: bool,
        sym: str,
    ) -> Tuple[bool, bool]:
        order_result: Optional[Dict[str, Any]] = None
        execution_error: Optional[Exception] = None
        try:
            if self._strategy._execution_adapter:
                order_result = await self._strategy._execution_adapter.execute(
                    self._strategy,
                    data.symbol,
                    data,
                    decision,
                    account_snapshot,
                )
            else:
                order_result = await self._strategy.adapter.execute_decision(
                    data.symbol,
                    data,
                    decision,
                    account_snapshot.raw,
                )
        except Exception as exc:
            execution_error = exc
        if execution_error is not None:
            raise execution_error
        if order_result:
            executed.append(order_result)
            if manager:
                manager.record_execution(
                    data.symbol,
                    decision,
                    order_result,
                )
            await self._strategy._notify_execution(data.symbol, decision, order_result)
            if manager and action_upper == "CLOSE":
                manager.remove(data.symbol)
                context.positions.pop(data.symbol, None)
            return True, False
        if manager and action_upper in {"ADJUST_SL", "ADJUST_TP", "TRAIL_ON", "TRAIL_OFF"}:
            manager.record_execution(
                data.symbol,
                decision,
                {},
            )
            updated_state = manager.get(data.symbol) if manager else None
            if updated_state:
                context.positions[data.symbol] = updated_state
        filled_qty = ensure_float(order_result.get("filled_qty") if order_result else None)
        if manager and used_provisional and (filled_qty is None or filled_qty <= 0):
            manager.remove(sym)
            context.positions.pop(sym, None)
            return False, False
        return True, False

    async def _apply_exit_rules(
        self,
        strategy: "AITradingStrategy",
        exit_engine,
        data: SymbolData,
        position,
        allowed_actions: Sequence[str],
        manager,
        execution_guard,
        account_snapshot,
        exchange_position,
        all_decisions: List[Dict[str, Any]],
        executed: List[Dict[str, Any]],
        context: StrategyContext,
    ) -> bool:
        if not exit_engine:
            return False
        ctx = ExitContext(
            symbol=data.symbol,
            position=position,
            price=data.last_price,
            timestamp=float(data.last_ts or utc_now().timestamp()),
            account_summary=account_snapshot.summary or {},
            symbol_meta=data.meta,
            payload=data.payload,
        )
        decisions = exit_engine.evaluate(ctx)
        if not decisions:
            return False

        executed_any = False
        allowed_set = {item.upper() for item in allowed_actions}
        for exit_decision in decisions:
            payload = exit_decision.to_decision_payload()
            payload.setdefault("sym", data.symbol)
            payload.setdefault("symbol", data.symbol)
            payload.setdefault("source", "auto_exit")

            if execution_guard:
                valid, guard_reason = execution_guard.validate_decision(
                    data.symbol,
                    payload,
                    account_snapshot,
                    position,
                    allowed_set,
                )
                if not valid:
                    strategy.logger.info(
                        "Auto-exit guard blocked decision: sym=%s action=%s rule=%s reason=%s",
                        data.symbol,
                        payload.get("action"),
                        payload.get("rule"),
                        guard_reason,
                    )
                    continue
                sized_amount = execution_guard.calc_position_size(
                    payload,
                    account_snapshot,
                    data,
                    position,
                )
                if payload.get("amount") is None and sized_amount > 0:
                    payload["amount"] = sized_amount

            strategy.logger.info(
                "Auto exit decision triggered: sym=%s action=%s amount=%s rule=%s reason=%s",
                data.symbol,
                payload.get("action"),
                payload.get("amount"),
                payload.get("rule"),
                payload.get("reason"),
            )

            await strategy._notify_exit_decision(payload)

            updated_position = manager.upsert(data.symbol, exchange_position or {}, payload)
            context.positions[data.symbol] = updated_position

            manager.update_risk_cap(
                data.symbol,
                account_snapshot.summary or {},
                multiplier=ensure_float(data.meta.get("multiplier"), None)
                if isinstance(data.meta, dict)
                else None,
            )

            order_result = None
            try:
                if strategy._execution_adapter:
                    order_result = await strategy._execution_adapter.execute(
                        strategy,
                        data.symbol,
                        data,
                        payload,
                        account_snapshot,
                    )
                else:
                    order_result = await strategy.adapter.execute_decision(
                        data.symbol,
                        data,
                        payload,
                        account_snapshot.raw,
                    )
            except Exception:
                strategy.logger.exception("Auto exit execution failed for %s", data.symbol)
                continue

            manager.record_execution(data.symbol, payload, order_result or {})
            all_decisions.append(payload)
            meta = payload.get("meta") or {}
            if meta.get("be_activate"):
                strategy.logger.info(
                    "Break-even stop activated for %s: new_sl=%s move=%s",
                    data.symbol,
                    payload.get("sl"),
                    meta.get("move_distance"),
                )
                telegram_service = getattr(strategy, "_telegram_service", None)
                if telegram_service:
                    try:
                        await telegram_service.notify_break_even_activation(
                            symbol=data.symbol,
                            payload=payload,
                            metadata=meta,
                        )
                    except Exception:
                        strategy.logger.exception(
                            "Failed to send break-even activation alert for %s", data.symbol
                        )
            if meta.get("trail_activate"):
                strategy.logger.info(
                    "Trailing activated for %s: new_sl=%s move=%s",
                    data.symbol,
                    payload.get("sl"),
                    meta.get("move_distance"),
                )
                telegram_service = getattr(strategy, "_telegram_service", None)
                if telegram_service:
                    try:
                        await telegram_service.notify_trailing_activation(
                            symbol=data.symbol,
                            payload=payload,
                            metadata=meta,
                        )
                    except Exception:
                        strategy.logger.exception("Failed to send trailing activation alert for %s", data.symbol)
            if order_result:
                executed.append(order_result)
                await strategy._notify_execution(data.symbol, payload, order_result)
            if str(payload.get("action", "")).upper() == "CLOSE":
                manager.remove(data.symbol)
                context.positions.pop(data.symbol, None)
            context.last_processed[data.symbol] = data.last_ts
            strategy._schedule_policy.record_success(strategy, data.symbol, data, payload)
            executed_any = True

        if executed_any:
            manager.save()
        return executed_any

    @staticmethod
    def _fallback_allowed_actions(position: Optional[object]) -> Set[str]:
        amount = 0.0
        if position is not None:
            try:
                amount = abs(float(getattr(position, "amount", 0.0)))
            except (TypeError, ValueError):
                amount = 0.0
        if amount <= 0:
            return {"BUY", "SELL", "HOLD", "WAIT", "ADJUST_SL", "ADJUST_TP"}
        return {"HOLD", "WAIT", "CLOSE", "REDUCE", "PARTIAL_TP", "ADD", "ADJUST_SL", "ADJUST_TP"}

    @staticmethod
    def _provisional_position_payload(
        decision: Dict[str, Any],
        data: SymbolData,
        current_position,
    ) -> Optional[Dict[str, Any]]:
        action = str(decision.get("action") or decision.get("act") or "").upper()
        if action not in {"BUY", "SELL", "ADD"}:
            return None
        try:
            current_amount = abs(float(getattr(current_position, "amount", 0.0)))
        except (TypeError, ValueError):
            current_amount = 0.0
        if current_amount > 0:
            return None

        amount = ensure_float(decision.get("amount"), 0.0)
        if amount <= 0:
            return None

        price = ensure_float(decision.get("entry"), None)
        if price is None or price <= 0:
            price = ensure_float(decision.get("price"), None)
        if price is None or price <= 0:
            price = ensure_float(data.last_price, 0.0)
        if price <= 0:
            return None

        side = "long" if action in {"BUY", "ADD"} else "short"

        return {
            "amount": amount,
            "entry_price": price,
            "avg_price": price,
            "mark": data.last_price,
            "side": side,
        }
