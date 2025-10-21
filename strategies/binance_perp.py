"""AI strategy wrapper for Binance perpetuals using the generic AI engine."""

from __future__ import annotations

import base64
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..adapters.binance_perp import BinancePerpetualAdapter
from ..core.defaults import DefaultAIEnvelopeAdapter
from ..engine.strategy import AITradingStrategy
from ..services.telegram import StrategyTelegramService
from ..core.interfaces import AccountSnapshot, SymbolData
from ..utils import ensure_float, format_ts, utc_now
from .common.decision_utils import normalize_decision_payload


def _summarize_execution_history(history: Sequence[Any]) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        if summary:
            snapshot = {
                "ts": entry.get("timestamp"),
                "action": summary.get("action"),
                "reason": summary.get("reason"),
                "side": summary.get("side"),
                "requested": summary.get("requested_qty"),
                "filled": summary.get("filled_qty"),
                "avg": summary.get("avg_price"),
            }
            summarized.append({k: v for k, v in snapshot.items() if v is not None})
            continue

        decision = entry.get("decision") if isinstance(entry.get("decision"), dict) else {}
        order = entry.get("order") if isinstance(entry.get("order"), dict) else {}
        filled = (
            order.get("filled")
            or order.get("executedQty")
            or order.get("executed_quantity")
            or order.get("qty")
            or entry.get("qty")
        )
        side = order.get("side") or entry.get("side")
        summarized.append(
            {
                "ts": entry.get("timestamp"),
                "action": entry.get("action"),
                "reason": decision.get("reason"),
                "side": side,
                "filled": filled,
            }
        )
    return summarized


class BinancePerpEnvelopeAdapter(DefaultAIEnvelopeAdapter):
    def encode_request(
        self,
        strategy: "AITradingStrategy",
        payload: Dict[str, Any],
        *,
        account: AccountSnapshot,
        symbols: Sequence[SymbolData],
    ) -> Dict[str, Any]:
        encoded = deepcopy(payload)
        result = super().encode_request(
            strategy,
            encoded,
            account=account,
            symbols=symbols,
        )

        history_conf = strategy.config.get("history", {})
        ai_conf = strategy.config.get("ai") or {}
        send_indicators = bool(ai_conf.get("send_indicators", True))
        timeframe_entries = history_conf.get("timeframes", [])
        tf_order: List[str] = []
        bars_map: Dict[str, int] = {}
        for entry in timeframe_entries:
            tf_text = str(entry.get("tf") or entry.get("timeframe") or "").lower()
            if not tf_text:
                continue
            if tf_text not in tf_order:
                tf_order.append(tf_text)
            bars_value = entry.get("bars")
            try:
                bars_int = int(bars_value)
            except (TypeError, ValueError):
                bars_int = 0
            if bars_int > 0:
                bars_map[tf_text] = bars_int

        def fmt_number(value: Any, *, signed: bool = False) -> str:
            if value is None:
                return "-"
            try:
                num = float(value)
            except (TypeError, ValueError):
                return str(value)
            abs_val = abs(num)
            if abs_val >= 100:
                decimals = 2
            elif abs_val >= 10:
                decimals = 3
            elif abs_val >= 1:
                decimals = 4
            else:
                decimals = 5
            fmt = f"{num:+.{decimals}f}" if signed else f"{num:.{decimals}f}"
            fmt = fmt.rstrip("0").rstrip(".")
            if signed and not fmt.startswith(("+", "-")):
                fmt = "+" + fmt
            return fmt or ("+0" if signed else "0")

        def clean_text(value: Any) -> str:
            if value is None:
                return "-"
            text = str(value)
            return text.replace("|", "/").replace("\n", " ").strip() or "-"

        account_lines: List[str] = []
        summary_source = account.summary or {}
        account_fields = [
            ("净值", "net_value"),
            ("可用保证金", "margin_free"),
            ("已用保证金", "margin_used"),
            ("持仓名", "positions"),
            ("总仓位名义", "total_pos_notional_value"),
        ]
        if isinstance(summary_source, dict):
            for label, key in account_fields:
                value = summary_source.get(key)
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(item) for item in value)
                account_lines.append(f"{label}|{clean_text(value)}")

        lines: List[str] = []
        if account_lines:
            lines.append("## Account")
            lines.append("项目|数值")
            lines.extend(account_lines)

        pairs = result.get("pairs")
        chart_map: Dict[str, List[str]] = {}

        if isinstance(pairs, list):
            for item in pairs:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("sym") or item.get("symbol") or "").upper()
                if not symbol:
                    continue
                item.pop("metadata", None)
                context = item.get("context") if isinstance(item.get("context"), dict) else {}
                if isinstance(context, dict):
                    context.pop("bounds", None)
                    context.pop("timeframes", None)
                indicators_source = item.get("indicators") if send_indicators else {}
                if isinstance(indicators_source, dict):
                    context.setdefault("indicators", indicators_source)
                else:
                    if not send_indicators:
                        context.pop("indicators", None)
                    else:
                        context.setdefault("indicators", {})
                chart_images = item.pop("chart_images", None)
                if chart_images:
                    chart_map[symbol] = list(chart_images)

                lines.append("")
                lines.append(f"## {symbol}")

                position_payload = item.get("position") if isinstance(item.get("position"), dict) else None
                summary_history: List[Dict[str, Any]] = []
                if position_payload:
                    history = position_payload.get("execution_history")
                    if isinstance(history, list):
                        summary_history = _summarize_execution_history(history)
                    position_payload.pop("execution_history", None)

                lines.append("### Position")
                if position_payload:
                    pos_table = ["字段|数值"]
                    pos_table.append(f"side|{clean_text(position_payload.get('side'))}")
                    pos_table.append(f"qty|{fmt_number(position_payload.get('amount'))}")
                    pos_table.append(f"entry|{fmt_number(position_payload.get('entry'))}")
                    pos_table.append(f"mark|{fmt_number(position_payload.get('mark'))}")
                    pos_table.append(f"upnl|{fmt_number(position_payload.get('upnl'), signed=True)}")
                    pos_table.append(f"sl|{fmt_number(position_payload.get('sl'))}")
                    targets = position_payload.get("targets")
                    if isinstance(targets, list) and targets:
                        pos_table.append("targets|" + "/".join(fmt_number(t) for t in targets))
                    trailing = position_payload.get("trailing")
                    if isinstance(trailing, dict) and trailing:
                        trail_kind = trailing.get("kind", "")
                        trail_multi = trailing.get("multi")
                        trail_anchor = trailing.get("anc") or trailing.get("anchor")
                        trail_text = f"{trail_kind.upper()}×{fmt_number(trail_multi)}@{trail_anchor or '-'}"
                        if trailing.get("step"):
                            trail_text += f" step={fmt_number(trailing.get('step'))}"
                        pos_table.append(f"trailing|{trail_text}")
                    tp_lock = position_payload.get("tp_lock")
                    if isinstance(tp_lock, dict) and tp_lock:
                        lock_text = ", ".join(
                            f"{key}={fmt_number(value)}" for key, value in tp_lock.items() if value is not None
                        )
                        if lock_text:
                            pos_table.append(f"tp_lock|{lock_text}")
                    if position_payload.get("rr") is not None:
                        pos_table.append(f"rr|{fmt_number(position_payload.get('rr'))}")
                    if position_payload.get("confidence") is not None:
                        pos_table.append(f"confidence|{fmt_number(position_payload.get('confidence'))}")
                    lines.extend(pos_table)
                    reason_text = position_payload.get("reason")
                    if reason_text:
                        lines.append(f"entry_reason: {clean_text(reason_text)}")
                else:
                    lines.append("字段|数值")
                    lines.append("状态|空仓")

                recent_execs = summary_history if summary_history else []
                if recent_execs:
                    lines.append("### Execution history")
                    lines.append("时间|动作|数量|均价|理由")
                    for entry in recent_execs:
                        lines.append(
                            f"{clean_text(entry.get('ts'))}|"
                            f"{clean_text(entry.get('action'))}|"
                            f"{fmt_number(entry.get('filled'))}|"
                            f"{fmt_number(entry.get('avg'))}|"
                            f"{clean_text(entry.get('reason'))}"
                        )

                indicators = context.get("indicators") if isinstance(context, dict) else {}
                if not isinstance(indicators, dict):
                    indicators = {}
                ema_raw = indicators.get("ema")
                rsi_raw = indicators.get("rsi")
                macd_raw = indicators.get("macd")
                atr_raw = indicators.get("atr")

                ema_map = ema_raw if isinstance(ema_raw, dict) else {}
                rsi_map = rsi_raw if isinstance(rsi_raw, dict) else {}
                macd_map = macd_raw if isinstance(macd_raw, dict) else {}
                atr_map = atr_raw if isinstance(atr_raw, dict) else {}

                tf_seen: List[str] = []
                for key in ema_map.keys():
                    tf_name = key.split("_", 1)[0]
                    if tf_name not in tf_seen:
                        tf_seen.append(tf_name)
                for tf_name in rsi_map.keys():
                    if tf_name not in tf_seen:
                        tf_seen.append(tf_name)
                for tf_name in macd_map.keys():
                    if tf_name not in tf_seen:
                        tf_seen.append(tf_name)
                for tf_name in atr_map.keys():
                    if tf_name not in tf_seen:
                        tf_seen.append(tf_name)

                ordered_tfs = []
                for tf in tf_order:
                    if tf in tf_seen:
                        ordered_tfs.append(tf)
                for tf in tf_seen:
                    if tf not in ordered_tfs:
                        ordered_tfs.append(tf)

                if ordered_tfs:
                    lines.append("### Indicators")
                    lines.append("周期|EMA|RSI|MACD状态|MACD/Signal|ATR")
                    for tf in ordered_tfs:
                        ema_entries: List[str] = []
                        for key, values in ema_map.items():
                            if not isinstance(values, list):
                                continue
                            parts = key.split("_", 1)
                            if parts[0] != tf:
                                continue
                            last_val = values[-1] if values else None
                            period = parts[1] if len(parts) > 1 else ""
                            label = period or key
                            ema_entries.append(f"{label}:{fmt_number(last_val)}")
                        ema_text = ", ".join(ema_entries) if ema_entries else "-"

                        rsi_values = rsi_map.get(tf)
                        rsi_last = rsi_values[-1] if isinstance(rsi_values, list) and rsi_values else None

                        macd_entry = macd_map.get(tf) if isinstance(macd_map.get(tf), dict) else {}
                        macd_state = macd_entry.get("state") if isinstance(macd_entry, dict) else None
                        macd_values = macd_entry.get("macd") if isinstance(macd_entry, dict) else None
                        macd_last = macd_values[-1] if isinstance(macd_values, list) and macd_values else None
                        signal_values = macd_entry.get("signal") if isinstance(macd_entry, dict) else None
                        signal_last = signal_values[-1] if isinstance(signal_values, list) and signal_values else None

                        atr_value = atr_map.get(tf)

                        lines.append(
                            f"{tf}|{ema_text}|{fmt_number(rsi_last)}|"
                            f"{clean_text(macd_state)}|{fmt_number(macd_last)}/{fmt_number(signal_last)}|"
                            f"{fmt_number(atr_value)}"
                        )

                kline_keys = [
                    key for key in item.keys() if isinstance(key, str) and key.startswith("kline_")
                ]
                if kline_keys:
                    lines.append("### Latest klines")
                    for key in sorted(kline_keys):
                        bars = item.get(key)
                        if not isinstance(bars, list) or not bars:
                            continue
                        tf_name = key.replace("kline_", "")
                        tf_lower = tf_name.lower()
                        bars_limit = bars_map.get(tf_lower, 0)
                        if not bars_limit or bars_limit <= 0:
                            bars_limit = len(bars)
                        n_recent = min(len(bars), bars_limit)
                        recent = bars[-n_recent:]
                        lines.append(f"{tf_name} (last {n_recent})")
                        lines.append("ts|o|h|l|c|v")
                        for idx, bar in enumerate(recent):
                            if not isinstance(bar, (list, tuple)) or len(bar) < 5:
                                continue
                            remaining = n_recent - idx - 1
                            if remaining == 0:
                                label = f"{tf_name}[0]"
                            else:
                                label = f"{tf_name}[-{remaining}]"
                            label = bar[5]
                            open_val = fmt_number(bar[0])
                            high_val = fmt_number(bar[1])
                            low_val = fmt_number(bar[2])
                            close_val = fmt_number(bar[3])
                            vol_val = fmt_number(bar[4])
                            lines.append(f"{label}|{open_val}|{high_val}|{low_val}|{close_val}|{vol_val}")

        final_text = "\n".join(line for line in lines if line is not None).strip()
        compact_result: Dict[str, Any] = {"text": final_text or "-"}
        if chart_map:
            compact_result["_chart_images"] = chart_map
        return compact_result

    def decode_response(
        self,
        strategy: "AITradingStrategy",
        response: Any,
        *,
        payload: Dict[str, Any],
        account: AccountSnapshot,
        symbols: Sequence[SymbolData],
    ) -> Any:
        decoded = super().decode_response(
            strategy,
            response,
            payload=payload,
            account=account,
            symbols=symbols,
        )
        if not isinstance(decoded, list):
            return decoded

        normalized: List[Dict[str, Any]] = []
        for entry in decoded:
            if not isinstance(entry, dict):
                continue
            normalized_entry = normalize_decision_payload(entry)
            if normalized_entry:
                normalized.append(normalized_entry)

        return normalized


class BinancePerpTelegramService(StrategyTelegramService):
    """Telegram helper combining command routing and status formatting."""

    def __init__(
        self,
        strategy: "AIBinancePerpStrategy",
        *,
        command_prefix: str,
        help_text: str,
    ) -> None:
        super().__init__(
            strategy,
            command_prefix=command_prefix,
            help_text=help_text,
        )

    async def handle_message(self, strategy: "AIBinancePerpStrategy", message: Dict[str, Any]) -> bool:  # type: ignore[override]
        if strategy is not self.strategy:
            return False
        if not strategy._is_authorized_sender(message):
            return False

        text = str(message.get("text") or "").strip()
        if not text.startswith(self.command_prefix):
            return await super().handle_message(strategy, message)

        parts = text.split()
        command = parts[1].lower() if len(parts) > 1 else "status"
        chat_id = str(message.get("chat_id") or message.get("chat") or "")

        if command == "status":
            summary = await self.strategy_status()
            if chat_id and summary:
                await strategy.send_telegram_message(summary, chat_id=chat_id, parse_mode="Markdown")
            return True
        if command == "portfolio":
            summary = await self.portfolio_status()
            if chat_id and summary:
                await strategy.send_telegram_message(summary, chat_id=chat_id, parse_mode="Markdown")
            return True
        if command in {"buy", "sell"}:
            if not chat_id:
                return True
            if len(parts) < 4:
                await strategy.send_telegram_message(
                    "用法：/ai_perp buy SYMBOL AMOUNT [PRICE]",
                    chat_id=chat_id,
                )
                return True
            symbol = parts[2]
            amount_str = parts[3]
            price_str = parts[4] if len(parts) > 4 else None
            await self.handle_manual_trade_command(
                chat_id=chat_id,
                side="BUY" if command == "buy" else "SELL",
                symbol=symbol,
                amount_str=amount_str,
                price_str=price_str,
            )
            return True
        if command == "close":
            if not chat_id:
                return True
            if len(parts) < 3:
                await strategy.send_telegram_message(
                    "用法：/ai_perp close SYMBOL",
                    chat_id=chat_id,
                )
                return True
            symbol = parts[2]
            await self.handle_close_position_command(chat_id=chat_id, symbol=symbol)
            return True
        if command == "decisions":
            symbol = parts[2] if len(parts) > 2 else None
            report = await self.render_decision_history(symbol)
            if chat_id and report:
                await strategy.send_telegram_message(report, chat_id=chat_id, parse_mode="Markdown")
            elif chat_id:
                await strategy.send_telegram_message("暂无相关决策记录", chat_id=chat_id)
            return True

        return await super().handle_message(strategy, message)

    async def strategy_status(self) -> Optional[str]:
        strategy = self.strategy
        lines: List[str] = ["*AI Strategy 状态*"]
        watchlist = (
            ", ".join(self._format_symbol(sym) for sym in strategy._symbols)
            if strategy._symbols
            else "-"
        )
        lines.append(f"• watchlist: `{self._escape_markdown(watchlist)}`")
        if strategy._last_run_time:
            lines.append(f"• last run: {format_ts(strategy._last_run_time.timestamp())}")
        if strategy._last_error:
            lines.append(f"• last error: {self._escape_markdown(strategy._last_error)}")
        if strategy._latest_decisions:
            lines.append("\n*Recent Decisions*")
            for decision in strategy._latest_decisions:
                sym = decision.get("symbol") or decision.get("sym")
                sym_fmt = self._format_symbol(sym)
                action = decision.get("action") or "?"
                amount = decision.get("amount") or decision.get("qty")
                qty_text = f" qty={amount}" if amount is not None else ""
                action_md = self._escape_markdown(action)
                reason_md = self._escape_markdown(decision.get("reason") or "-")
                lines.append(f"- `{sym_fmt}` **{action_md}**{qty_text}\n  reason: {reason_md}")
        else:
            lines.append("\n*Recent Decisions*")
            lines.append("- 无记录")
        return "\n".join(lines)

    async def portfolio_status(self) -> Optional[str]:
        strategy = self.strategy
        try:
            raw_account_state = await strategy.adapter.fetch_account_state()
            snapshot = strategy._account_formatter.format(raw_account_state)
            strategy._last_account_snapshot = snapshot
        except Exception:
            snapshot = strategy._last_account_snapshot
        if not snapshot:
            return "尚未获取账户信息"
        summary = snapshot.summary or {}
        net_value = (
            summary.get("net_value")
            or summary.get("equity")
            or summary.get("total_equity")
        )
        margin_free = summary.get("margin_free") or summary.get("available")
        margin_used = summary.get("margin_used") or summary.get("margin")
        lines: List[str] = ["*Portfolio 状态*"]
        if net_value is not None:
            lines.append(f"• net value: `{self._format_number(net_value)}`")
        if margin_free is not None:
            lines.append(f"• margin free: `{self._format_number(margin_free)}`")
        if margin_used is not None:
            lines.append(f"• margin used: `{self._format_number(margin_used)}`")
        total_notional = summary.get("total_pos_notional_value")
        if total_notional is not None:
            lines.append(f"• total notional: `{self._format_number(total_notional)}`")

        positions = snapshot.raw.get("positions") or []
        if positions:
            lines.append("\n*Positions*")
            for pos in positions:
                sym = self._format_symbol(pos.get("symbol"))
                qty = pos.get("contracts")
                if qty is None:
                    qty = pos.get("amount") or pos.get("size")
                try:
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    qty_val = 0.0
                side = pos.get("side", "long")
                side_label = "" if side == "long" else "-"
                qty_text = f"{abs(qty_val):g}" if qty is not None else "-"
                avg_price = (
                    pos.get("entry_price")
                    or pos.get("avg_price")
                    or pos.get("entry")
                )
                avg_text = self._format_price(avg_price)
                upnl = (
                    pos.get("unrealized_pnl")
                    or pos.get("upnl")
                    or pos.get("unrealized")
                )
                upnl_text = self._format_number(upnl)
                current_price = (
                    pos.get("mark")
                    or pos.get("mark_price")
                    or pos.get("price")
                    or pos.get("last_price")
                )
                current_text = self._format_price(current_price)
                normalized_symbol = strategy.normalize_symbol(pos.get("symbol") or "")
                position_record = strategy.position_manager.get(normalized_symbol)
                sl_value: Optional[float] = None
                risk_flag = ""
                if position_record:
                    sl_value = position_record.sl
                    if getattr(position_record, "reached_risk_cap", False):
                        risk_flag = "(capped)"
                sl_text = self._format_price(sl_value) if sl_value is not None else "-"
                lines.append(
                    f"- `{sym}{risk_flag}`: `{side_label}{qty_text}` @ `{avg_text}`|`{current_text}` sl: `{sl_text}` upnl: `{upnl_text}`"
                )
        else:
            lines.append("\n*Positions*\n- 当前无持仓")

        return "\n".join(lines)

    async def handle_manual_trade_command(
        self,
        *,
        chat_id: str,
        side: str,
        symbol: str,
        amount_str: str,
        price_str: Optional[str],
    ) -> None:
        strategy = self.strategy
        normalized = strategy.normalize_symbol(symbol)
        try:
            amount = float(amount_str)
        except (TypeError, ValueError):
            await strategy.send_telegram_message("数量格式错误", chat_id=chat_id)
            return
        if amount <= 0:
            await strategy.send_telegram_message("数量必须大于0", chat_id=chat_id)
            return

        price: Optional[float] = None
        if price_str:
            try:
                price = float(price_str)
            except (TypeError, ValueError):
                await strategy.send_telegram_message("价格格式错误", chat_id=chat_id)
                return
            if price <= 0:
                await strategy.send_telegram_message("价格必须大于0", chat_id=chat_id)
                return

        if normalized not in strategy._symbols:
            await strategy._add_symbol(normalized)

        result = await self._submit_manual_order(
            symbol=normalized,
            side=side,
            amount=amount,
            price=price,
            reduce_only=False,
            reason=f"manual {side.lower()} command",
        )

        if result is None:
            await strategy.send_telegram_message("下单失败，请查看日志", chat_id=chat_id)

    async def handle_close_position_command(self, *, chat_id: str, symbol: str) -> None:
        strategy = self.strategy
        normalized = strategy.normalize_symbol(symbol)
        account_state = await strategy.adapter.fetch_account_state()
        snapshot = strategy._account_formatter.format(account_state)
        strategy._last_account_snapshot = snapshot

        position = self._find_position(snapshot.raw.get("positions") or [], normalized)
        if not position:
            await strategy.send_telegram_message(
                f"`{self._format_symbol(normalized)}` 当前无持仓",
                chat_id=chat_id,
                parse_mode="Markdown",
            )
            return

        qty = ensure_float(position.get("contracts") or position.get("amount") or position.get("size"))
        side_flag = str(position.get("side") or "long").lower()
        if qty == 0:
            await strategy.send_telegram_message(
                f"`{self._format_symbol(normalized)}` 当前仓位为0",
                chat_id=chat_id,
                parse_mode="Markdown",
            )
            return

        close_side = "BUY" if side_flag == "short" else "SELL"
        result = await self._submit_manual_order(
            symbol=normalized,
            side=close_side,
            amount=abs(qty),
            price=None,
            reduce_only=True,
            reason="manual close command",
        )

        if result is None:
            await strategy.send_telegram_message("平仓指令失败，请查看日志", chat_id=chat_id)

    async def render_decision_history(self, symbol: Optional[str]) -> Optional[str]:
        import json

        strategy = self.strategy
        decisions_path = strategy.state_store.decisions_path

        if not decisions_path.exists():
            return None

        try:
            records = json.loads(decisions_path.read_text())
        except Exception:
            return None
        if not isinstance(records, list):
            return None

        records = [r for r in records if isinstance(r, dict)]
        if not records:
            return None

        records.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)

        normalized = strategy.normalize_symbol(symbol) if symbol else None
        symbols_set = {strategy.normalize_symbol(s) for s in strategy._symbols}

        def _format_orders(sym: str, *, limit: int, bullet: str, indent: str) -> List[str]:
            position_record = strategy.position_manager.get(sym)
            history = position_record.execution_history if position_record else None
            if not isinstance(history, list) or not history:
                return []
            lines: List[str] = []
            for item in list(history)[-limit:][::-1]:
                if not isinstance(item, dict):
                    continue
                ts = self._format_timestamp(item.get("timestamp"))
                summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
                action = summary.get("action") or item.get("action") or "?"
                qty = summary.get("filled_qty") or summary.get("requested_qty")
                qty_text = self._format_number(qty) if qty is not None else "-"
                price = summary.get("avg_price")
                price_text = f" price={self._format_price(price)}" if price is not None else ""
                action_md = self._escape_markdown(action)
                lines.append(f"{bullet}`{ts}` **{action_md}** qty={qty_text}{price_text}")
                reason = summary.get("reason") or item.get("decision", {}).get("reason")
                if reason:
                    reason_md = self._escape_markdown(reason)
                    lines.append(f"{indent}reason: {reason_md}")
            return lines

        if normalized:
            entries: List[str] = []
            collected = 0
            for record in records:
                ts = self._format_timestamp(record.get("timestamp"))
                for entry in record.get("decisions", []):
                    sym = strategy.normalize_symbol(entry.get("symbol") or entry.get("sym") or "")
                    if sym != normalized:
                        continue
                    action = entry.get("action") or "?"
                    amount = entry.get("amount") or entry.get("qty")
                    qty_text = f" qty={self._format_number(amount)}" if amount is not None else ""
                    action_md = self._escape_markdown(action)
                    reason_md = self._escape_markdown(entry.get("reason") or "-")
                    entries.append(f"- `{ts}` **{action_md}**{qty_text}\n  reason: {reason_md}")
                    collected += 1
                    if collected >= 10:
                        break
                if collected >= 10:
                    break
            if not entries:
                return None
            lines: List[str] = [f"*最近 AI 决策* (`{normalized}`)"]
            lines.extend(entries)
            order_lines = _format_orders(normalized, limit=5, bullet="- ", indent="  ")
            lines.append("\n*订单历史 (最近5条)*")
            if order_lines:
                lines.extend(order_lines)
            else:
                lines.append("- 暂无执行记录")
            return "\n".join(lines)

        latest_record = records[0]
        decisions = latest_record.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            return None
        timestamp = self._format_timestamp(latest_record.get("timestamp"))
        response_ids = latest_record.get("response_ids")
        response_text = ""
        if isinstance(response_ids, list) and response_ids:
            response_text = " rsp=" + self._escape_markdown(",".join(map(str, response_ids)))
        lines = [f"*最近一次 AI 调用* (`{timestamp}`){response_text}"]
        for entry in decisions:
            sym_raw = entry.get("symbol") or entry.get("sym") or ""
            sym = strategy.normalize_symbol(sym_raw)
            action = entry.get("action") or "?"
            amount = entry.get("amount") or entry.get("qty")
            qty_text = f" qty={self._format_number(amount)}" if amount is not None else ""
            action_md = self._escape_markdown(action)
            reason_md = self._escape_markdown(entry.get("reason") or "-")
            sym_fmt = self._format_symbol(sym_raw)
            lines.append(f"- `{sym_fmt}` **{action_md}**{qty_text}\n  reason: {reason_md}")
            order_lines = _format_orders(sym, limit=1, bullet="  • ", indent="    ") if sym in symbols_set else []
            if order_lines:
                lines.extend(order_lines)

        if len(lines) == 1:
            return None
        lines.append("\n(使用 `/ai_perp decisions SYMBOL` 查看某个标的的详细历史)")
        return "\n".join(lines)

    async def _submit_manual_order(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float],
        reduce_only: bool,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        strategy = self.strategy
        result = await strategy.adapter.place_manual_order(
            symbol,
            side=side,
            amount=amount,
            price=price,
            reduce_only=reduce_only,
        )
        if not result:
            return None

        order = result.get("order") or {}
        filled = ensure_float(
            order.get("filled")
            or order.get("executedQty")
            or order.get("executed_quantity")
            or result.get("filled_qty")
            or 0.0
        )
        avg_price = (
            order.get("avgPrice")
            or order.get("average")
            or order.get("average_price")
            or order.get("price")
        )
        avg_price_float = ensure_float(avg_price)
        result.update(
            {
                "requested_qty": amount,
                "filled_qty": filled,
                "avg_price": avg_price_float if avg_price_float else None,
                "remaining_qty": max(0.0, amount - filled),
                "sub_orders": [order] if order else [],
                "side": result.get("side") or side,
            }
        )

        
        decision = {
            "action": f"MANUAL_{side.upper()}",
            "reason": reason,
            "symbol": symbol,
            "amount": amount,
        }
        entry_timestamp = format_ts(utc_now().timestamp())

        position_manager = strategy.position_manager
        normalized_symbol = strategy.normalize_symbol(symbol)
        existing_position = position_manager.get(normalized_symbol)
        placeholder_amount = filled if filled is not None and filled > 0 else amount
        placeholder_price = avg_price_float or price or result.get("avg_price") or 0.0
        if existing_position is None and placeholder_amount and placeholder_amount > 0:
            placeholder_payload = {
                "symbol": normalized_symbol,
                "amount": placeholder_amount,
                "entry": placeholder_price,
                "mark": placeholder_price,
                "timestamp": entry_timestamp,
                "side": side.lower(),
            }
            position_manager.upsert(normalized_symbol, placeholder_payload, decision=decision, update_revision=False)
            new_position = position_manager.get(normalized_symbol)
            if new_position:
                new_position.timestamp = entry_timestamp

        await strategy._notify_execution(symbol, decision, result)
        position_manager.record_execution(symbol, decision, result)
        await self._maybe_archive_plan_after_manual(symbol, decision, entry_timestamp)
        strategy.position_manager.save()
        return result

    def _find_position(self, positions: Sequence[Dict[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
        strategy = self.strategy
        normalized = strategy.normalize_symbol(symbol)
        for pos in positions:
            sym = strategy.normalize_symbol(pos.get("symbol") or "")
            if sym == normalized:
                return pos
        return None

    async def _maybe_archive_plan_after_manual(self, symbol: str, decision: Dict[str, Any], entry_ts: str) -> None:
        strategy = self.strategy
        position_manager = strategy.position_manager
        try:
            state = await strategy.adapter.fetch_account_state()
        except Exception:
            return
        positions = state.get("positions") or []
        normalized = strategy.normalize_symbol(symbol)
        position = self._find_position(positions, normalized)
        amount_value = ensure_float(position.get("amount") or position.get("contracts")) if position else 0.0
        if position and amount_value and amount_value > 0:
            position_manager.upsert(normalized, position, decision=decision)
            position_record = position_manager.get(normalized)
            if position_record and decision.get("action", "").upper().startswith("MANUAL_"):
                history = position_record.execution_history
                if history and isinstance(history[-1], dict):
                    history[-1]["timestamp"] = entry_ts
        else:
            position_manager.remove(normalized)
        position_manager.save()

    def _format_symbol(self, symbol: Optional[str]) -> str:
        if not symbol:
            return "?"
        normalized = self.strategy.normalize_symbol(symbol)
        if normalized.endswith("USDT"):
            normalized = normalized[:-4]
        return normalized

    def _format_price(self, price: Optional[float]) -> str:
        if price is None:
            return "-"
        try:
            price = float(price)
        except (TypeError, ValueError):
            return "-"
        abs_price = abs(price)
        if abs_price >= 100:
            decimals = 2
        elif abs_price >= 10:
            decimals = 3
        elif abs_price >= 1:
            decimals = 4
        elif abs_price >= 0.1:
            decimals = 5
        elif abs_price >= 0.01:
            decimals = 6
        elif abs_price >= 0.001:
            decimals = 7
        else:
            decimals = 8
        formatted = f"{price:.{decimals}f}".rstrip("0").rstrip(".")
        return formatted or "0"

    def _format_number(self, value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"

    def _format_timestamp(self, value: Any) -> str:
        from datetime import datetime

        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return format_ts(float(value))
        text = str(value).strip()
        if not text:
            return "-"
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            return format_ts(dt.timestamp())
        except Exception:
            return self._escape_markdown(text)

DEFAULT_PROMPT_TEMPLATE = """
你是相对激进的Binance USDT-M永续合约的交易判断顾问。每1小时收盘后调用一次。

职责边界：
- 只做市场结构解读与高层动作建议；确定性逻辑（杠杆/仓位/冷却/下单等）由外部系统执行。

输入包含：当前时间、每个 symbol 的多周期K线数据及指标摘要、账户摘要、该 symbol 的仓位摘要以及K线图。
long only:{long_only} 市场波动较大，可尽量放宽止损止盈规则，趋势不明朗信号不明确时尽量不开仓
risk preference:{risk_preference}

为每个需要评估的 symbol 输出一条 JSON 对象，字段：
- sym: string                 // 交易对，如 "BTCUSDT"
- pb: string                  // 推荐 playbook: "TREND_PULLBACK" | "BREAKOUT_RET" | "RANGE_FADE" | "VOL_SQUEEZE" | "MOM_REVERSION" | "MEAN_REVERSION_BANDS" | "LIQUIDITY_SWEEP_RET" | "RISK_OFF"
- act: string                 // "WAIT" | "BUY" | "SELL" | "ADD" | "REDUCE" | "CLOSE" | "PARTIAL_TP" | "ADJUST_SL" | "ADJUST_TP" | "TRAIL_ON" | "TRAIL_OFF" | "HOLD"
- conf: number                // 0..1，win_rate estimation
- rr: number                  // 预估风险收益比（Reward/Risk）
- rsn: string                 // ≤140字，结构化理由（趋势/区间、关键位、量价、动量/波动）
- sl: number                 // 合理的失效/止损价（swing 位/形态边界等）
- tgts: [number..]          // max 2 targets

// 首次开仓或当前仓位无sl和tp策略时，补充以下策略参数建议：
- tp_pol: "fixed"(tp use tgts) | "trailing"(tp use trailing) | "hybrid"(tp use tgts+trailing)
- tp_lock?: {{
    sl_to_be: number          // 达到多少 R 后，把 SL 抬到 BE（break-even）
    trail_after: number       // 达到多少 R 后，启动 trailing
  }}
- trail?: {{                   // trailing 参数（仅当 tp_pol ∈ {{"trailing","hybrid"}} 时）
    kind: "atr" | "pct" | "chandelier"
    multi?: number            // ATR 倍数（atr/chandelier）
    dist?: number             // 百分比距离，如 0.02 表示 2%（pct）
    anc?: "close" | "high_low" // 锚点
    step?: number             // 止损“阶梯式”锁盈步进（价格变动累计多少再上调一次）
    floor?: number | null     // 止损“硬底/硬顶”（多头为最低止损线）
  }}

// 提示：请结合行情、仓位、风险参数选择更合理的sl/tgts/tp_pol等参数（保守/中性/激进）

输出要求：
- 返回合法 JSON 数组；不得包含多余文本。
- 如果仓位reached_risk_cap为真，不加仓（ADD）
- 如果加仓（ADD），需给出完整的风险参数：sl, tgts, tp_pol, tp_lock & trail
- 首次开仓/当前有仓位但无sl/tp策略→输出含sl/tp策略的完整字段
- WAIT时仅需{{sym, act, rsn}}无需其他字段
- 其他场景仅需固定字段 {{sym, act, rsn}}+需要更新的字段
- 根据给出的数据选择合适的交易时间周期及对应tp/sl参数
- 尽量只参与正EV的交易
- 仓位名义价值小于{min_reduce_notional}时，不允许PARTIAL_TP/REDUCE
"""


class AIBinancePerpStrategy(AITradingStrategy):
    """Concrete AI strategy for Binance perpetuals."""

    MESSAGE_HEADER = "AI-BNPERP"
    BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
    CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
    ADMIN_IDS_ENV = "TELEGRAM_ADMIN_IDS"

    def __init__(self, config: Dict[str, Any]):
        adapter = BinancePerpetualAdapter(config)
        constraints_conf = config.get("constraints") or {}

        long_only = "是" if constraints_conf.get("long_only", False) else "否"
        risk_preference = constraints_conf.get("risk_preference", "medium") # high, low, medium
        min_reduce_notional = constraints_conf.get("min_reduce_notional", 1000)
        prompt = DEFAULT_PROMPT_TEMPLATE.format(
            long_only=long_only,
            risk_preference=risk_preference,
            min_reduce_notional=min_reduce_notional
        )
        history_conf = config.get("history") or {}
        timeframes_conf = history_conf.get("timeframes")
        configured_timeframes: List[str] = []
        if isinstance(timeframes_conf, list):
            for entry in timeframes_conf:
                if not isinstance(entry, dict):
                    continue
                tf_value = str(entry.get("timeframe") or "").strip()
                if not tf_value:
                    continue
                tf_lower = tf_value.lower()
                if tf_lower not in configured_timeframes:
                    configured_timeframes.append(tf_lower)
        if not configured_timeframes:
            configured_timeframes = ["1h", "4h"]

        ai_conf = config.get("ai") or {}
        chart_conf = dict(history_conf.get("chart") or {})
        ai_chart_conf = ai_conf.get("chart") or {}
        if ai_chart_conf:
            chart_conf.update(ai_chart_conf)

        self._chart_enabled = bool(chart_conf.get("enabled", False))
        intervals_conf = chart_conf.get("intervals")
        if intervals_conf:
            intervals: List[str] = []
            for value in intervals_conf:
                tf_str = str(value or "").strip()
                if not tf_str:
                    continue
                tf_lower = tf_str.lower()
                if tf_lower not in intervals:
                    intervals.append(tf_lower)
        else:
            intervals = list(configured_timeframes)
        if not intervals:
            intervals = list(configured_timeframes) or ["1h", "4h"]
        self._chart_intervals = tuple(intervals)

        self._chart_lookback = int(chart_conf.get("lookback", 100) or 100)
        chart_dir = chart_conf.get("output_dir") or os.path.join("rockstock", "charts")
        self._chart_dir = Path(chart_dir)
        if self._chart_enabled:
            self._chart_dir.mkdir(parents=True, exist_ok=True)
        strategy_key = (
            config.get("strategy_key")
            or (config.get("strategy") or {}).get("key")
            or "ai_binance_perp"
        )
        default_symbols: Sequence[str] = config.get("default_symbols") or ["BTCUSDT", "ETHUSDT"]
        telegram_conf = config.get("telegram") or {}
        command_prefix = str(telegram_conf.get("command_prefix", "/ai"))

        help_text = (
            "指令格式："
            "/ai_perp status | /ai_perp portfolio | /ai_perp add SYMBOL | /ai_perp remove SYMBOL | "
            "/ai_perp plan [SYMBOL] | /ai_perp run | /ai_perp decisions [SYMBOL] | "
            "/ai_perp buy SYMBOL AMOUNT [PRICE] | /ai_perp sell SYMBOL AMOUNT [PRICE] | /ai_perp close SYMBOL"
        )

        super().__init__(
            config,
            adapter=adapter,
            strategy_key=strategy_key,
            default_prompt=prompt,
            default_symbols=default_symbols,
            envelope_adapter=BinancePerpEnvelopeAdapter(),
            telegram_service_factory=lambda s: BinancePerpTelegramService(
                s,
                command_prefix=command_prefix,
                help_text=help_text,
            ),
        )

    # ------------------------------------------------------------------
    # Prompt & payload overrides
    # ------------------------------------------------------------------

    def build_prompt_messages(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
        chart_map = payload.pop("_chart_images", {})
        image_paths: List[str] = []
        if isinstance(chart_map, dict):
            for paths in chart_map.values():
                if isinstance(paths, (list, tuple)):
                    image_paths.extend(str(path) for path in paths)

        text_block = payload.get("text") if isinstance(payload, dict) else None
        remaining_keys = [key for key in payload.keys() if key not in {"text"}] if isinstance(payload, dict) else []
        if isinstance(text_block, str) and not remaining_keys:
            user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": text_block}]
        else:
            user_content = [
                {
                    "type": "input_text",
                    "text": json.dumps(payload, ensure_ascii=self._payload_ensure_ascii),
                }
            ]

        for path in image_paths:
            try:
                user_content.append(
                    {"type": "input_image", "image_url": self._encode_image(Path(path))}
                )
            except Exception:
                self.logger.exception("Failed to encode chart image: %s", path)

        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self._system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    @staticmethod
    def _encode_image(path: Path) -> str:
        with path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

__all__ = ["AIBinancePerpStrategy", "BinancePerpEnvelopeAdapter"]
