"""Unified Telegram service handling commands and notifications."""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..utils import ensure_float, format_ts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..engine.strategy import AITradingStrategy


class StrategyTelegramService:
    """Default Telegram helper that processes commands and status updates."""

    DEFAULT_HELP = (
        "指令格式：/ai status | /ai portfolio | /ai add SYMBOL | /ai remove SYMBOL | /ai plan [SYMBOL] | /ai run"
    )

    def __init__(
        self,
        strategy: "AITradingStrategy",
        *,
        command_prefix: str = "/ai",
        help_text: Optional[str] = None,
    ) -> None:
        self._strategy = strategy
        self._logger = strategy.logger
        self.command_prefix = command_prefix
        self._help_text = help_text or self.DEFAULT_HELP
        self._extra_handlers: List[Any] = []

    # ------------------------------------------------------------------
    # Command registration
    # ------------------------------------------------------------------

    def add_handler(self, handler: Any) -> None:
        if handler and handler not in self._extra_handlers:
            self._extra_handlers.append(handler)

    @property
    def strategy(self) -> "AITradingStrategy":
        return self._strategy

    async def _dispatch_extras(self, strategy: "AITradingStrategy", message: Dict[str, Any]) -> bool:
        for handler in list(self._extra_handlers):
            try:
                if hasattr(handler, "handle_message"):
                    result = handler.handle_message(strategy, message)
                else:
                    result = handler(strategy, message)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:  # pragma: no cover - defensive guard
                self._logger.exception("Telegram handler failed: %s", handler)
                continue
            if result:
                return True
        return False

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        strategy: "AITradingStrategy",
        message: Dict[str, Any],
    ) -> bool:
        if strategy is not self._strategy:
            return False
        if not strategy._is_authorized_sender(message):
            return False

        text = str(message.get("text") or "").strip()
        if not text.startswith(self.command_prefix):
            return await self._dispatch_extras(strategy, message)

        chat_id = str(message.get("chat_id") or message.get("chat") or "")
        parts = text.split()
        command = parts[1].lower() if len(parts) > 1 else "status"
        arg = parts[2:] if len(parts) > 2 else []

        if command == "status":
            summary = await strategy.strategy_status()
            if summary:
                await strategy.send_telegram_message(summary, chat_id=chat_id)
        elif command == "portfolio":
            summary = await strategy.portfolio_status()
            if summary:
                await strategy.send_telegram_message(summary, chat_id=chat_id)
        elif command == "add" and arg:
            symbol = strategy.normalize_symbol(arg[0])
            added = await strategy._add_symbol(symbol)
            msg = f"已添加监控 {symbol}" if added else f"{symbol} 已在监控列表或添加失败"
            await strategy.send_telegram_message(msg, chat_id=chat_id)
        elif command in {"remove", "del"} and arg:
            symbol = strategy.normalize_symbol(arg[0])
            removed = await strategy._remove_symbol(symbol)
            msg = f"已移除 {symbol}" if removed else f"未找到 {symbol}"
            await strategy.send_telegram_message(msg, chat_id=chat_id)
        elif command == "plan":
            if arg:
                symbol = strategy.normalize_symbol(arg[0])
                await strategy._send_plan(symbol, chat_id)
            else:
                await strategy._send_all_plans(chat_id)
        elif command in {"run", "scan", "trigger"}:
            strategy._force_event.set()
            await strategy.send_telegram_message("已触发即时AI评估", chat_id=chat_id)
        else:
            await strategy.send_telegram_message(self._help_text, chat_id=chat_id)
        return True

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    async def strategy_status(self) -> Optional[str]:
        strategy = self._strategy
        lines = ["AI Strategy 状态:"]
        lines.append("监控列表: " + ", ".join(strategy._symbols))
        if strategy._last_run_time:
            lines.append(f"最近执行: {format_ts(strategy._last_run_time.timestamp())}")
        if strategy._last_error:
            lines.append(f"最近错误: {strategy._last_error}")
        if strategy._latest_decisions:
            lines.append("最近AI决策:")
            for decision in strategy._latest_decisions:
                sym = decision.get("sym")
                action = decision.get("action")
                qty = decision.get("qty")
                reason = decision.get("reason") or ""
                lines.append(f"  - {sym}: {action} qty={qty} {reason}")
        else:
            lines.append("尚无AI决策")
        return "\n".join(lines)

    async def portfolio_status(self) -> Optional[str]:
        strategy = self._strategy
        if not strategy._last_account_snapshot:
            return "尚未获取账户信息"
        snapshot = strategy._last_account_snapshot
        summary = snapshot.summary or {}
        raw = snapshot.raw
        lines = ["账户概览:"]
        net_value = summary.get("net_value") or summary.get("equity") or raw.get("net_value")
        margin_free = (
            summary.get("margin_free")
            or summary.get("available")
            or raw.get("margin_free")
            or raw.get("available")
        )
        margin_used = summary.get("margin_used") or summary.get("margin") or raw.get("margin_used")
        if net_value is not None:
            lines.append(f"• 净值: {net_value:.2f} USDT")
        if margin_free is not None:
            lines.append(f"• 可用保证金: {margin_free:.2f} USDT")
        if margin_used is not None:
            lines.append(f"• 已用保证金: {margin_used:.2f} USDT")
        positions = snapshot.positions or raw.get("positions") or summary.get("positions") or []
        if positions:
            lines.append("• 持仓:")
            if isinstance(positions, dict):
                iterator = ((sym, pos) for sym, pos in positions.items())
            else:
                iterator = (
                    (pos.get("symbol") or pos.get("sym") or "?", pos)
                    for pos in positions
                )
            for sym, pos in iterator:
                qty = pos.get("contracts")
                if qty is None:
                    amount = float(pos.get("amount", 0.0))
                    side = str(pos.get("side") or "").lower()
                    qty = amount if side != "short" else -amount
                side_text = "多" if qty >= 0 else "空"
                avg = pos.get("entry_price") or pos.get("entry") or 0.0
                lines.append(f"  - {sym}: {side_text} {abs(qty)} @ {avg}")
        else:
            lines.append("• 当前无持仓")
        return "\n".join(lines)

    async def notify_execution(
        self,
        symbol: str,
        decision: Dict[str, Any],
        order_result: Dict[str, Any],
    ) -> None:
        strategy = self._strategy
        requested = ensure_float(order_result.get("requested_qty") or decision.get("amount"))
        filled = ensure_float(order_result.get("filled_qty") or order_result.get("qty") or 0.0)
        avg_price = order_result.get("avg_price")

        action = str(decision.get("action") or decision.get("act") or "")
        action_upper = action.upper()

        header = "*Execution Summary*"
        info_line_1 = (
            f"• symbol: `{symbol}` | action: `{action}` | side: `{order_result.get('side')}`"
        )
        info_line_2 = (
            f"• requested: `{requested:.6f}` | filled: `{filled:.6f}`"
            if requested is not None
            else f"• filled: `{filled:.6f}`"
        )
        price_text = f"{avg_price:.6f}" if avg_price is not None else "-"
        message_lines = [header, info_line_1, info_line_2, f"• avg price: `{price_text}`"]

        if action_upper == "CLOSE":
            pnl_line = self._build_realized_pnl_line(symbol, order_result, decision)
            if pnl_line:
                message_lines.append(pnl_line)

        reason = decision.get("reason")
        if reason:
            message_lines.append("")
            message_lines.append(f"_reason_: {self._escape_markdown(reason)}")

        message = "\n".join(message_lines)
        try:
            await strategy.send_telegram_message(message, parse_mode="Markdown")
        except Exception:
            self._logger.exception("Failed to send order notification")

    async def notify_exit_decision(self, decision: Dict[str, Any]) -> None:
        strategy = self._strategy

        symbol = self._escape_markdown(decision.get("sym") or decision.get("symbol") or "-")
        action = self._escape_markdown(decision.get("action") or "-")
        amount = ensure_float(decision.get("amount"), None)
        reason = decision.get("reason")
        rule = decision.get("rule")

        message_lines = [
            "*Auto Exit Decision*",
            f"• symbol: `{symbol}`",
            f"• action: `{action}`",
        ]

        if amount is not None:
            message_lines.append(f"• amount: `{amount:.6f}`")

        if decision.get("sl") is not None:
            message_lines.append(f"• stop loss: `{ensure_float(decision.get('sl'), 0.0):.6f}`")

        targets = decision.get("targets")
        if targets:
            try:
                targets_text = ", ".join(f"{ensure_float(t, 0.0):.6f}" for t in targets)
            except Exception:
                targets_text = ", ".join(str(t) for t in targets)
            message_lines.append(f"• targets: `{self._escape_markdown(targets_text)}`")

        if rule:
            message_lines.append(f"• rule: `{self._escape_markdown(rule)}`")

        metadata = decision.get("meta") or decision.get("metadata")
        if metadata:
            try:
                meta_text = json.dumps(metadata, ensure_ascii=False, default=str)
            except Exception:
                meta_text = str(metadata)
            message_lines.append(f"• meta: `{self._escape_markdown(meta_text)}`")

        if reason:
            message_lines.append("")
            message_lines.append(f"_reason_: {self._escape_markdown(reason)}")

        trailing = decision.get("trailing")
        if trailing:
            try:
                trailing_text = json.dumps(trailing, ensure_ascii=False, default=str)
            except Exception:
                trailing_text = str(trailing)
            message_lines.append("")
            message_lines.append(f"_trailing_: `{self._escape_markdown(trailing_text)}`")

        tp_policy = decision.get("tp_policy")
        if tp_policy:
            message_lines.append(f"_tp policy_: `{self._escape_markdown(str(tp_policy))}`")

        message = "\n".join(message_lines)

        try:
            await strategy.send_telegram_message(message, parse_mode="Markdown")
        except Exception:
            self._logger.exception("Failed to send exit decision notification")

    async def notify_trailing_activation(
        self,
        *,
        symbol: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        strategy = self._strategy

        sl_value = ensure_float(payload.get("sl"), None)
        price_value = ensure_float(metadata.get("price"), None)
        move_distance = ensure_float(metadata.get("move_distance"), None)

        message_lines = [
            "*Trailing Stop Activated*",
            f"• symbol: `{self._escape_markdown(symbol)}`",
        ]

        if sl_value is not None:
            message_lines.append(f"• new SL: `{sl_value:.6f}`")
        if price_value is not None:
            message_lines.append(f"• price: `{price_value:.6f}`")
        if move_distance is not None:
            message_lines.append(f"• move distance: `{move_distance:.6f}`")

        reason = payload.get("reason")
        if reason:
            message_lines.append("")
            message_lines.append(f"_reason_: {self._escape_markdown(reason)}")

        message = "\n".join(message_lines)
        try:
            await strategy.send_telegram_message(message, parse_mode="Markdown")
        except Exception:
            self._logger.exception("Failed to send trailing activation notification")

    async def notify_break_even_activation(
        self,
        *,
        symbol: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        strategy = self._strategy

        sl_value = ensure_float(payload.get("sl"), None)
        price_value = ensure_float(metadata.get("price"), None)
        move_distance = ensure_float(metadata.get("move_distance"), None)

        message_lines = [
            "*Break-even Stop Activated*",
            f"• symbol: `{self._escape_markdown(symbol)}`",
        ]

        if sl_value is not None:
            message_lines.append(f"• new SL: `{sl_value:.6f}`")
        if price_value is not None:
            message_lines.append(f"• price: `{price_value:.6f}`")
        if move_distance is not None:
            message_lines.append(f"• move distance: `{move_distance:.6f}`")

        reason = payload.get("reason")
        if reason:
            message_lines.append("")
            message_lines.append(f"_reason_: {self._escape_markdown(reason)}")

        message = "\n".join(message_lines)
        try:
            await strategy.send_telegram_message(message, parse_mode="Markdown")
        except Exception:
            self._logger.exception("Failed to send break-even activation notification")

    @staticmethod
    def _escape_markdown(value: Any) -> str:
        if value is None:
            return "-"
        text = str(value).replace("\n", " ")
        replacements = {
            "\\": "\\\\",
            "`": "\`",
            "*": "\*",
            "_": "\_",
            "[": "\[",
            "]": "\]",
            "(": "\(",
            ")": "\)",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _build_realized_pnl_line(
        self,
        symbol: str,
        order_result: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Optional[str]:
        strategy = self._strategy
        position = strategy.position_manager.get(symbol)
        if not position:
            return None

        entry = ensure_float(getattr(position, "entry", None), None)
        exit_price = ensure_float(
            order_result.get("avg_price")
            or order_result.get("price")
            or decision.get("price"),
            None,
        )
        filled = ensure_float(
            order_result.get("filled_qty")
            or order_result.get("qty")
            or decision.get("amount"),
            None,
        )
        if entry is None or exit_price is None or filled is None:
            return None

        qty = abs(filled)
        if qty <= 0:
            return None

        side = (getattr(position, "side", "long") or "long").lower()
        if side == "short":
            pnl = (entry - exit_price) * qty
        else:
            pnl = (exit_price - entry) * qty

        pnl_text = self._format_decimal(pnl)
        return f"• realized pnl: `{pnl_text}`"

    @staticmethod
    def _format_decimal(value: float) -> str:
        abs_value = abs(value)
        if abs_value >= 1000:
            decimals = 2
        elif abs_value >= 100:
            decimals = 3
        elif abs_value >= 10:
            decimals = 4
        elif abs_value >= 1:
            decimals = 5
        elif abs_value >= 0.1:
            decimals = 6
        elif abs_value >= 0.01:
            decimals = 7
        else:
            decimals = 8
        formatted = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"


__all__ = ["StrategyTelegramService"]
