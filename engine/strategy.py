"""Core framework for AI-driven trading strategies."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from strategy.base import PROJECT_ROOT, StrategyBase

from ..utils import ensure_float, first_float, format_ts, utc_now
from ..core.defaults import DefaultAccountFormatter
from ..core.interfaces import (
    AccountFormatter,
    AccountSnapshot,
    AIEnvelopeAdapter,
    BatchPlanner,
    ExecutionAdapter,
    MarketAdapter,
    PayloadBuilder,
    SchedulePolicy,
    SymbolData,
)
from ..core.position import Position
from ..core.context import StrategyContext
from ..exit_engine import (
    RuleExitEngine,
    StopLossExitRule,
    TargetExitRule,
    BreakEvenStopRule,
    TrailingExitRule,
)
from ..services.scheduler import DefaultSchedulePolicy
from ..services.state import StrategyStateStore
from ..services.telegram import StrategyTelegramService
from ..services.position_manager import PositionManager
from ..services.ai_client import OpenAIClient, AIClient
from ..filters import PreExecutionFilter
from ..execution import ExecutionGuard
from .pipeline import DecisionPipeline


class AITradingStrategy(StrategyBase):
    """Generic AI-driven trading strategy orchestrator."""

    DEFAULT_STRATEGY_KEY = "ai_strategy"

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        adapter: MarketAdapter,
        strategy_key: Optional[str] = None,
        default_prompt: Optional[str] = None,
        default_symbols: Optional[Sequence[str]] = None,
        command_handler: Optional[Any] = None,
        command_handlers: Optional[Sequence[Any]] = None,
        ai_client:  Optional[AIClient] = None,
        account_formatter: Optional[AccountFormatter] = None,
        schedule_policy: Optional[SchedulePolicy] = None,
        batch_planner: Optional[BatchPlanner] = None,
        payload_builder: Optional[PayloadBuilder] = None,
        execution_adapter: Optional[ExecutionAdapter] = None,
        envelope_adapter: Optional[AIEnvelopeAdapter] = None,
        state_store_factory: Optional[Callable[["AITradingStrategy"], StrategyStateStore]] = None,
        telegram_service_factory: Optional[
            Callable[["AITradingStrategy"], StrategyTelegramService]
        ] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.adapter = adapter
        self.strategy_key = (
            strategy_key
            or config.get("strategy_key")
            or (config.get("strategy") or {}).get("key")
            or self.DEFAULT_STRATEGY_KEY
        )

        self._storage_root = PROJECT_ROOT / "dbfiles"
        self._state_store = self._init_state_store(state_store_factory)
        self._storage_dir = self._state_store.storage_dir
        self._decisions_dir = self._state_store.storage_dir
        self._execution_log_path = self._state_store.execution_log_path

        ai_conf = config.get("ai") or {}
        history_conf = config.get("history") or {}

        self._ai_model = str(ai_conf.get("model", "gpt-4o-mini"))
        self._ai_temperature = float(ai_conf.get("temperature", 0.1))
        self._ai_max_tokens = int(ai_conf.get("max_tokens", 0))

        self._auto_run_ai = bool(ai_conf.get("run_on_start", True))

        self._system_prompt = self._load_system_prompt(ai_conf, default_prompt)
        self._payload_ensure_ascii = bool(ai_conf.get("payload_ascii", False))

        self._ai_client: Optional[AIClient] = self._init_ai_client(ai_conf, ai_client)

        self._risk_conf = config.get("risk") or {}
        self._constraints = config.get("constraints") or {}
        features_conf = config.get("features") or {}
        self._init_risk_controls(features_conf)

        self._evaluation_seconds = self._compute_evaluation_seconds(history_conf)
        self._poll_delay_seconds = int(config.get("poll_delay", 30))
        self._force_event: asyncio.Event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._exit_task: Optional[asyncio.Task] = None

        self._init_services(
            account_formatter,
            schedule_policy,
            batch_planner,
            payload_builder,
            execution_adapter,
            envelope_adapter,
            command_handler,
            command_handlers,
            telegram_service_factory,
        )

        self._load_symbol_set(default_symbols or config.get("default_symbols") or ["BTCUSDT", "ETHUSDT"])
        self._schedule_policy.prepare(self)
        self._last_processed: Dict[str, int] = self._state_store.load_last_processed()

        self._last_account_snapshot: Optional[AccountSnapshot] = None
        self._latest_decisions: List[Dict[str, Any]] = []
        self._last_run_time: Optional[datetime] = None
        self._last_error: Optional[str] = None

        self._adapter_ready = False

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def _build_context(self) -> StrategyContext:
        return StrategyContext(
            strategy_key=self.strategy_key,
            config=self.config,
            symbols=list(self._symbols),
            positions=self._positions,
            last_processed=self._last_processed,
            account_snapshot=self._last_account_snapshot,
            latest_decisions=list(self._latest_decisions),
            last_run_time=self._last_run_time,
            last_error=self._last_error,
            extras={"adapter_ready": self._adapter_ready},
            position_manager=self._position_manager,
        )

    def _apply_context(self, context: StrategyContext) -> None:
        self._symbols = list(context.symbols)
        manager_positions = self._position_manager.positions
        if context.position_manager and context.position_manager is not self._position_manager:
            self._position_manager = context.position_manager
            manager_positions = self._position_manager.positions
        if context.positions is not manager_positions:
            manager_positions.clear()
            manager_positions.update(context.positions)
        self._positions = manager_positions
        self._last_processed = context.last_processed
        self._last_account_snapshot = context.account_snapshot
        self._latest_decisions = list(context.latest_decisions)
        self._last_run_time = context.last_run_time
        self._last_error = context.last_error
        if context.extras:
            self._adapter_ready = context.extras.get("adapter_ready", self._adapter_ready)

    def _init_state_store(
        self,
        state_store_factory: Optional[Callable[["AITradingStrategy"], StrategyStateStore]],
    ) -> StrategyStateStore:
        if state_store_factory:
            return state_store_factory(self)
        return StrategyStateStore(
            base_dir=self._storage_root,
            strategy_key=self.strategy_key,
            normalize_symbol=self.normalize_symbol,
            logger=self.logger,
        )

    def _init_ai_client(self, ai_conf: Dict[str, Any], ai_client: Optional[AIClient]) -> Optional[AIClient]:
        if ai_client is not None:
            return ai_client
        api_key_env = str(ai_conf.get("api_key_env", "OPENAI_API_KEY"))
        ai_api_key = str(ai_conf.get("api_key") or os.getenv(api_key_env, "")).strip()
        if not ai_api_key:
            self.logger.warning("AI API key missing，自动交易将无法调用AI")
            return None
        try:
            return OpenAIClient(
                api_key=ai_api_key,
                model=self._ai_model,
                temperature=self._ai_temperature,
                max_tokens=self._ai_max_tokens,
                logger=self.logger,
            )
        except Exception:  # pragma: no cover - network/auth issues
            self.logger.exception("Failed to initialize OpenAI client")
            return None

    def _init_risk_controls(self, features_conf: Dict[str, Any]) -> None:
        self._risk_unit_pct = float(self._risk_conf.get("risk_unit_R_pct", 0) or 0.0)
        self._max_risk_total_pct = float(self._risk_conf.get("max_risk_total_R_pct", 0) or 0.0)
        self._pre_filter = PreExecutionFilter(
            cooldown_bars=self._constraints.get("cooldown_bars", 0),
            long_only=self._constraints.get("long_only", False),
            max_active_positions=self._constraints.get("max_active_positions"),
            min_reduce_notional=self._constraints.get("min_reduce_notional", 1000.0),
        )
        self._pre_filter_enabled = bool(features_conf.get("enable_rule_filters", False))
        guard_enabled = bool(features_conf.get("enable_execution_guard", False))
        self._execution_guard_enabled = guard_enabled
        self._execution_guard = (
            ExecutionGuard(
                min_rr=self._constraints.get("min_rr", 0.0),
                long_only=self._constraints.get("long_only", False),
                min_reduce_notional=float(self._constraints.get("min_reduce_notional", 1000.0)),
            )
            if guard_enabled
            else None
        )
        rule_names = features_conf.get(
            "exit_rules",
            [
                "StopLossExitRule",
                "TargetExitRule",
                "BreakEvenStopRule",
                "TrailingExitRule",
            ],
        )
        registry = {
            "StopLossExitRule": StopLossExitRule,
            "TargetExitRule": TargetExitRule,
            "BreakEvenStopRule": BreakEvenStopRule,
            "TrailingExitRule": TrailingExitRule,
        }
        rule_instances = [registry[name]() for name in rule_names if name in registry]
        self._exit_engine_enabled = bool(rule_instances)
        self._exit_engine = RuleExitEngine(rule_instances) if rule_instances else None
        exit_conf = features_conf.get("exit_engine") or {}
        self._exit_poll_interval = float(exit_conf.get("poll_interval_seconds", 60.0))
        pipeline_exit_rules_conf = features_conf.get("pipeline_exit_rules")
        if pipeline_exit_rules_conf is None:
            self._exit_rules_pipeline_enabled = True
        else:
            self._exit_rules_pipeline_enabled = bool(pipeline_exit_rules_conf)

    def _compute_evaluation_seconds(self, history_conf: Dict[str, Any]) -> int:
        evaluation_seconds = int(
            history_conf.get("evaluation_seconds")
            or history_conf.get("eval_seconds")
            or history_conf.get("evaluation_interval", 0)
            or history_conf.get("eval_interval", 0)
            or 0
        )
        if evaluation_seconds:
            return evaluation_seconds
        primary_tf = None
        timeframes_conf = history_conf.get("timeframes")
        if isinstance(timeframes_conf, list) and timeframes_conf:
            first_entry = timeframes_conf[0]
            if isinstance(first_entry, dict):
                primary_tf = str(first_entry.get("timeframe") or "")
        if not primary_tf:
            primary_tf = str(history_conf.get("primary_timeframe", "1h"))
        from ..utils import timeframe_to_seconds  # avoid circular import at top

        return timeframe_to_seconds(primary_tf) or 3600

    def _init_services(
        self,
        account_formatter: Optional[AccountFormatter],
        schedule_policy: Optional[SchedulePolicy],
        batch_planner: Optional[BatchPlanner],
        payload_builder: Optional[PayloadBuilder],
        execution_adapter: Optional[ExecutionAdapter],
        envelope_adapter: Optional[AIEnvelopeAdapter],
        command_handler: Optional[Any],
        command_handlers: Optional[Sequence[Any]],
        telegram_service_factory: Optional[Callable[["AITradingStrategy"], StrategyTelegramService]],
    ) -> None:
        self._account_formatter = account_formatter or DefaultAccountFormatter()
        self._schedule_policy = schedule_policy or DefaultSchedulePolicy(
            self._evaluation_seconds,
            self._poll_delay_seconds,
        )
        self._batch_planner = batch_planner
        self._payload_builder = payload_builder
        self._envelope_adapter = envelope_adapter
        self._execution_adapter = execution_adapter
        self._position_manager = PositionManager(
            self._state_store,
            json_safe=self._json_safe,
            logger=self.logger,
        )
        self._positions = self._position_manager.positions
        if telegram_service_factory:
            self._telegram_service = telegram_service_factory(self)
        else:
            self._telegram_service = StrategyTelegramService(self)

        if command_handlers:
            for handler in command_handlers:
                self._telegram_service.add_handler(handler)
        if command_handler:
            self._telegram_service.add_handler(command_handler)

        self._notifier = self._telegram_service
        self._decision_pipeline = DecisionPipeline(self)

    async def start(self) -> None:
        if not self._exit_engine_enabled or self._exit_task:
            return
        loop = asyncio.get_running_loop()
        self._exit_task = loop.create_task(self._exit_monitor_loop())

    def _load_symbol_set(self, default_symbols: Sequence[str]) -> None:
        config_symbols = self.config.get("symbols") or []
        normalized_config = {self.normalize_symbol(sym) for sym in config_symbols if sym}
        if normalized_config:
            normalized = normalized_config
        else:
            persisted_symbols = self._state_store.load_watchlist()
            seed_symbols: Iterable[str] = persisted_symbols if persisted_symbols else default_symbols
            normalized = {self.normalize_symbol(sym) for sym in seed_symbols if sym}
            if not normalized:
                normalized = {self.normalize_symbol(sym) for sym in default_symbols}
        self._symbols = sorted(normalized)
        self._state_store.save_watchlist(self._symbols)

    def _ensure_symbol_tracking(self, symbol: str) -> str:
        normalized = self.normalize_symbol(symbol)
        if normalized in self._symbols:
            return normalized
        self._symbols.append(normalized)
        self._symbols = sorted(set(self._symbols))
        self._state_store.save_watchlist(self._symbols)
        self._schedule_policy.ensure_symbol(self, normalized)
        self.logger.info("Discovered live position for %s; appended to watchlist", normalized)
        return normalized

    def normalize_symbol(self, symbol: str) -> str:
        return self.adapter.normalize_symbol(symbol)

    async def subscribe_telegram(self, bot_name: str, durable_name: str):
        """Override to start monitor loop and exit engine before subscribing."""
        await self.start()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        try:
            await super().subscribe_telegram(bot_name, durable_name)
        finally:
            if self._monitor_task:
                self._monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await self._monitor_task
            await self.close()

    async def close(self) -> None:
        await super().close()
        with contextlib.suppress(Exception):
            await self.adapter.close()
        if self._exit_task:
            self._exit_task.cancel()
            with contextlib.suppress(Exception):
                await self._exit_task

    @property
    def state_store(self) -> StrategyStateStore:
        return self._state_store

    @property
    def telegram_service(self) -> StrategyTelegramService:
        return self._telegram_service

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions

    @property
    def position_manager(self) -> PositionManager:
        return self._position_manager

    # ------------------------------------------------------------------
    # Telegram handlers
    # ------------------------------------------------------------------

    def register_command_handler(self, handler: Any) -> None:
        if self._telegram_service:
            self._telegram_service.add_handler(handler)

    async def process_tg_message(self, message_dict: Dict[str, Any]):
        handled = False
        if self._telegram_service:
            handled = await self._telegram_service.handle_message(self, message_dict)
        if not handled:
            await super().process_tg_message(message_dict)

    async def process_tg_callback(self, callback_dict):
        return

    async def portfolio_status(self) -> Optional[str]:
        return await self._notifier.portfolio_status()

    async def strategy_status(self) -> Optional[str]:
        return await self._notifier.strategy_status()

    # ------------------------------------------------------------------
    # Watchlist & persistence helpers
    # ------------------------------------------------------------------

    async def _add_symbol(self, symbol: str) -> bool:
        if symbol in self._symbols:
            return False
        self._symbols.append(symbol)
        self._symbols.sort()
        self._state_store.save_watchlist(self._symbols)
        self._schedule_policy.ensure_symbol(self, symbol)
        return True

    async def _remove_symbol(self, symbol: str) -> bool:
        if symbol not in self._symbols:
            return False
        self._symbols.remove(symbol)
        self._state_store.save_watchlist(self._symbols)
        position = self._position_manager.get(symbol)
        should_remove_position = True
        if position:
            try:
                if abs(float(position.amount)) > 0:
                    should_remove_position = False
            except (TypeError, ValueError):
                should_remove_position = True
        if should_remove_position:
            self._position_manager.remove(symbol)
            self._positions.pop(symbol, None)
            if symbol in self._last_processed:
                self._last_processed.pop(symbol, None)
            self._position_manager.save()
            self._state_store.save_last_processed(self._last_processed)
        else:
            self.logger.info(
                "Removed %s from watchlist but keeping open position until exit engine closes it",
                symbol,
            )
        self._schedule_policy.drop_symbol(self, symbol)
        return True


    def _json_safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return value

    async def _notify_execution(
        self,
        symbol: str,
        decision: Dict[str, Any],
        order_result: Dict[str, Any],
    ) -> None:
        await self._notifier.notify_execution(symbol, decision, order_result)

    async def _notify_exit_decision(self, decision: Dict[str, Any]) -> None:
        notifier = self._notifier
        if notifier and hasattr(notifier, "notify_exit_decision"):
            await notifier.notify_exit_decision(decision)

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        first_cycle = True
        while True:
            forced = self._force_event.is_set()
            if forced:
                self._force_event.clear()
            try:
                should_run = forced or not first_cycle or self._auto_run_ai
                if should_run:
                    await self._run_once(force=forced or first_cycle)
                    self._last_error = None
                    first_cycle = False
                else:
                    self.logger.info("AI auto-start disabled，刷新账户数据后等待手动触发 (/ai_perp run)")
                    await self._ensure_adapter()
                    raw_account_state = await self.adapter.fetch_account_state()
                    self._last_account_snapshot = self._account_formatter.format(raw_account_state)
                    first_cycle = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - runtime guard
                self._last_error = str(exc)
                self.logger.exception("AI strategy monitor loop error")
            wait_seconds = self._time_to_next_cycle()
            try:
                await asyncio.wait_for(self._force_event.wait(), timeout=wait_seconds)
            except asyncio.TimeoutError:
                continue

    def _time_to_next_cycle(self) -> float:
        return float(self._schedule_policy.next_run_delay(self, self._last_run_time))

    async def _ensure_adapter(self) -> None:
        if self._adapter_ready:
            return
        await self.adapter.setup()
        self._adapter_ready = True

    async def _run_once(self, *, force: bool = False) -> None:
        context = self._build_context()
        context = await self._decision_pipeline.run(context, force=force)
        self._apply_context(context)

    async def _exit_monitor_loop(self) -> None:
        self.logger.info(
            "Exit engine monitor started (interval=%.1fs)",
            self._exit_poll_interval,
        )
        try:
            while True:
                if not self._exit_engine_enabled:
                    await asyncio.sleep(self._exit_poll_interval)
                    continue

                try:
                    await self._ensure_adapter()
                    raw_account_state = await self.adapter.fetch_account_state()
                    account_snapshot = self._account_formatter.format(raw_account_state)
                    self._apply_risk_budget(account_snapshot)
                except Exception:
                    self.logger.exception("Exit engine failed to fetch account state")
                    await asyncio.sleep(self._exit_poll_interval)
                    continue

                self._last_account_snapshot = account_snapshot
                account_summary = account_snapshot.summary or {}
                manager = self._position_manager
                if not manager.positions:
                    await asyncio.sleep(self._exit_poll_interval)
                    continue

                context = self._build_context()
                context.account_snapshot = account_snapshot
                pre_filter = self._pre_filter
                pre_filter_enabled = self._pre_filter_enabled
                execution_guard = self._execution_guard if self._execution_guard_enabled else None
                exit_engine = self._exit_engine

                raw_positions = account_snapshot.raw.get("positions") if account_snapshot.raw else None
                exchange_positions: Dict[str, Any] = {}
                if isinstance(raw_positions, list):
                    for payload in raw_positions:
                        if not isinstance(payload, dict):
                            continue
                        symbol_text = payload.get("symbol") or payload.get("sym")
                        if not symbol_text:
                            continue
                        normalized_sym = self.normalize_symbol(str(symbol_text))
                        exchange_positions[normalized_sym] = payload

                positions_changed = False
                for sym, payload in exchange_positions.items():
                    updated = manager.upsert(sym, payload, update_revision=False)
                    if updated is None:
                        manager.remove(sym)
                        self._positions.pop(sym, None)
                        positions_changed = True
                        continue
                    manager.update_risk_cap(
                        sym,
                        account_summary,
                        multiplier=ensure_float(payload.get("multiplier"), None)
                        if isinstance(payload, dict)
                        else None,
                    )
                    try:
                        remaining_amt = abs(float(updated.amount))
                    except (TypeError, ValueError):
                        remaining_amt = 0.0
                    if remaining_amt <= 0:
                        manager.remove(sym)
                        self._positions.pop(sym, None)
                        positions_changed = True
                        continue
                    tracked = self._ensure_symbol_tracking(sym)
                    self._positions[tracked] = updated
                if positions_changed:
                    manager.save()

                local_decisions: List[Dict[str, Any]] = []
                local_orders: List[Dict[str, Any]] = []
                changes_made = False

                # Archive positions that no longer exist on the exchange (manual close)
                for sym in list(self._positions.keys()):
                    if sym in exchange_positions:
                        continue
                    position_record = manager.get(sym)
                    current_amount = ensure_float(getattr(position_record, "amount", 0.0)) if position_record else 0.0
                    if position_record and abs(current_amount) > 0:
                        self.logger.info("Detected manual close for %s; archiving position", sym)
                    if position_record:
                        manager.remove(sym)
                        positions_changed = True
                    self._positions.pop(sym, None)
                if positions_changed:
                    manager.save()

                history_conf = self.config.get("history") or {}
                fetch_concurrency_raw = history_conf.get("fetch_concurrency", 5)
                try:
                    fetch_concurrency = int(fetch_concurrency_raw)
                except (TypeError, ValueError):
                    fetch_concurrency = 5
                if fetch_concurrency <= 0:
                    fetch_concurrency = 1
                semaphore = asyncio.Semaphore(fetch_concurrency)

                async def _collect_exit_symbol(sym: str, pos: Optional[Position]) -> Tuple[str, Optional[SymbolData], Optional[Position]]:
                    if hasattr(self.adapter, "is_market_open"):
                        try:
                            market_open = await self.adapter.is_market_open(sym)
                        except Exception:
                            self.logger.exception("Exit engine market hours check failed for %s", sym)
                            market_open = True
                        if not market_open:
                            self.logger.debug("Exit engine skip %s: market closed", sym)
                            return sym, None, pos
                    try:
                        async with semaphore:
                            data_local = await self.adapter.collect_symbol_data(
                                sym,
                                account_snapshot.raw,
                                position=pos,
                                risk=self._risk_conf,
                                constraints=self._constraints,
                                use_chart=False,
                            )
                    except Exception:
                        self.logger.exception("Exit engine failed to collect symbol data for %s", sym)
                        return sym, None, pos
                    return sym, data_local, pos

                symbol_keys = list(self._positions.keys())
                tasks: List[asyncio.Task[Tuple[str, Optional[SymbolData], Optional[Position]]]] = [
                    asyncio.create_task(_collect_exit_symbol(sym, self._positions.get(sym)))
                    for sym in symbol_keys
                ]
                exit_results: List[Tuple[str, Optional[SymbolData], Optional[Position]]] = []
                if tasks:
                    gathered = await asyncio.gather(*tasks, return_exceptions=True)
                    for idx, result in enumerate(gathered):
                        sym = symbol_keys[idx]
                        if isinstance(result, Exception):
                            self.logger.exception("Exit engine symbol task failed for %s", sym, exc_info=result)
                            continue
                        exit_results.append(result)

                for sym, data, original_position in exit_results:
                    position = self._positions.get(sym) or original_position
                    if position is None:
                        continue
                    if data is None:
                        continue
                    self.logger.debug(
                        "Exit engine received symbol data: sym=%s last_price=%s last_ts=%s",
                        sym,
                        getattr(data, "last_price", None),
                        getattr(data, "last_ts", None),
                    )
                    data.position = position
                    last_exit_ts = manager.last_exit(sym)
                    if last_exit_ts is not None:
                        data.meta["last_exit_ts"] = last_exit_ts

                    allowed_actions: Optional[Set[str]] = None
                    if pre_filter:
                        account_summary = account_snapshot.summary or {}
                        allowed_actions = set(
                            pre_filter.get_allowed_actions(sym, position, account_summary)
                        )
                        if pre_filter_enabled:
                            skip, reason = pre_filter.should_skip_ai(
                                sym,
                                data,
                                account_snapshot,
                                position,
                            )
                            if skip:
                                self.logger.debug(
                                    "Exit engine pre-filter skip %s: %s",
                                    sym,
                                    reason,
                                )
                                continue
                    if not allowed_actions:
                        allowed_actions = DecisionPipeline._fallback_allowed_actions(position)
                    allowed_actions = {str(item).upper() for item in allowed_actions if item}
                    if not allowed_actions:
                        allowed_actions = {"HOLD"}
                    allowed_list = sorted(allowed_actions)

                    exit_executed = await self._decision_pipeline._apply_exit_rules(
                        self,
                        exit_engine,
                        data,
                        position,
                        allowed_list,
                        manager,
                        execution_guard,
                        account_snapshot,
                        exchange_positions.get(sym),
                        local_decisions,
                        local_orders,
                        context,
                    )

                    if exit_executed:
                        changes_made = True

                if local_decisions:
                    timestamp_iso = format_ts(utc_now().timestamp())
                    account_payload = self._json_safe(
                        account_snapshot.raw or account_snapshot.summary or {}
                    )
                    self.state_store.persist_decision(
                        timestamp_iso,
                        ["auto_exit"],
                        account_payload,
                        local_decisions,
                    )
                    context.latest_decisions = local_decisions

                if local_orders:
                    self.state_store.append_execution_log(
                        {"ts": format_ts(utc_now().timestamp()), "orders": local_orders}
                    )

                if changes_made:
                    self._apply_context(context)

                await asyncio.sleep(self._exit_poll_interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("Exit engine monitor loop error")
        finally:
            self._exit_task = None
            self.logger.info("Exit engine monitor stopped")

    # ------------------------------------------------------------------
    # AI invocation
    # ------------------------------------------------------------------

    async def _invoke_ai(self, payload: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        if not self._ai_client:
            raise RuntimeError("AI client 未初始化")

        messages = self.build_prompt_messages(payload)
        return await self._ai_client.invoke(messages, parser=self._parse_ai_response)

    def build_prompt_messages(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = self._system_prompt
        if not system_prompt:
            raise RuntimeError("System prompt 未配置")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=self._payload_ensure_ascii),
            },
        ]

    def _parse_ai_response(self, text: Optional[str]) -> Any:
        if not text:
            return []
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return []
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return []

    async def _send_plan(self, symbol: str, chat_id: str) -> None:
        position = self._position_manager.get(symbol)
        if not position:
            await self.send_telegram_message(f"{symbol} 暂无持仓记录", chat_id=chat_id)
            return
        payload = json.dumps(position.to_dict(), ensure_ascii=False, indent=2)
        await self.send_telegram_message(
            f"{symbol} Position:\n<pre>{payload}</pre>",
            chat_id=chat_id,
            parse_mode="HTML",
        )

    async def _send_all_plans(self, chat_id: str) -> None:
        if not self._position_manager.positions:
            await self.send_telegram_message("暂无任何持仓记录", chat_id=chat_id)
            return
        for symbol in sorted(self._position_manager.positions):
            await self._send_plan(symbol, chat_id)

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Prompt loader
    # ------------------------------------------------------------------

    def _load_system_prompt(self, ai_conf: Dict[str, Any], default_prompt: Optional[str]) -> str:
        prompt_conf = ai_conf.get("prompt") or {}
        system_prompt = prompt_conf.get("system") or ai_conf.get("system_prompt")
        prompt_path = prompt_conf.get("path") or ai_conf.get("prompt_path")
        if prompt_path:
            prompt_file = Path(prompt_path)
            if not prompt_file.is_absolute():
                prompt_file = PROJECT_ROOT / prompt_file
            try:
                system_prompt = prompt_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                self.logger.error("Prompt file not found: %s", prompt_file)
            except Exception:
                self.logger.exception("Failed to read prompt file: %s", prompt_file)
        if not system_prompt:
            system_prompt = default_prompt or "You are an execution router for trading strategies."
        return system_prompt

    # ------------------------------------------------------------------
    # Risk budget helpers
    # ------------------------------------------------------------------

    def _apply_risk_budget(self, snapshot: AccountSnapshot) -> None:
        summary = snapshot.summary if isinstance(snapshot.summary, dict) else {}

        net_value = self._resolve_net_value(snapshot, summary)

        risk_unit = ensure_float(self._risk_conf.get("risk_unit_R"), 0.0)
        if self._risk_unit_pct > 0 and net_value > 0:
            risk_unit = net_value * self._risk_unit_pct

        max_risk_total = ensure_float(self._risk_conf.get("max_risk_total_R"), 0.0)
        if self._max_risk_total_pct > 0 and net_value > 0:
            max_risk_total = net_value * self._max_risk_total_pct

        summary["risk_unit_R"] = round(risk_unit, 2) if risk_unit else 0.0
        summary["max_risk_total_R"] = round(max_risk_total, 2) if max_risk_total else 0.0

        snapshot.summary = summary

        self.logger.info(
            "Risk budget updated: net_value=%.2f risk_unit_R=%.2f max_risk_total_R=%.2f",
            net_value,
            summary.get("risk_unit_R", 0.0),
            summary.get("max_risk_total_R", 0.0),
        )

        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        risk_meta = metadata.setdefault("risk_budget", {})
        risk_meta.update(
            {
                "net_value": net_value,
                "risk_unit_R": summary.get("risk_unit_R", 0.0),
                "max_risk_total_R": summary.get("max_risk_total_R", 0.0),
            }
        )
        snapshot.metadata = metadata

    def _resolve_net_value(self, snapshot: AccountSnapshot, summary: Dict[str, Any]) -> float:
        keys = ("net_value", "equity", "total_equity", "balance", "cash")
        for container in (summary, snapshot.prompt_payload, snapshot.raw):
            if not isinstance(container, dict):
                continue
            numeric = first_float(container, keys, default=None, require_positive=True)
            if numeric is not None:
                return numeric
        return 0.0


__all__ = ["AITradingStrategy"]
