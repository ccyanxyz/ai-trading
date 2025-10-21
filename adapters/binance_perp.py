"""Market adapter implementations for Binance-based AI strategies."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ccxt.async_support as ccxt_async

from data.binance import BinanceProvider
from signals.indicators import atr, ema_series, macd_series, rsi_series
from ..utils import (
    ensure_float,
    normalize_symbol,
    timeframe_to_seconds,
    to_ccxt_symbol,
    trim_to_closed_bars,
    format_ts,
)
from ..core.position import Position
from ..core.interfaces import MarketAdapter, SymbolData
from ..core.position import Position
from .base import BaseMarketAdapter


def _format_bar(bar: Iterable[Any]) -> List[Any]:
    values = list(bar)
    if len(values) < 6:
        return values
    ts_sec = int(ensure_float(values[0], 0.0) / 1000)
    return [
        ensure_float(values[1]),
        ensure_float(values[2]),
        ensure_float(values[3]),
        ensure_float(values[4]),
        ensure_float(values[5]),
        ts_sec,
    ]


def _build_bounds(symbol: str, market: Optional[Dict[str, Any]], last_price: float) -> Dict[str, Any]:
    if not market:
        return {
            "qty_abs": [0.0, None],
            "lot_step": 1.0,
            "min_notional": None,
            "max_notional": None,
        }
    limits = market.get("limits", {})
    amount_limits = limits.get("amount", {})
    cost_limits = limits.get("cost", {})
    min_qty = ensure_float(amount_limits.get("min"), 0.0)
    max_raw = amount_limits.get("max")
    max_qty = ensure_float(max_raw, float("inf")) if max_raw is not None else float("inf")
    step = ensure_float(amount_limits.get("step"), 0.0) or market.get("contractSize") or 1.0
    min_notional = ensure_float(cost_limits.get("min"))
    max_notional = ensure_float(cost_limits.get("max"))
    if min_notional and last_price:
        min_qty = max(min_qty, min_notional / last_price)
    if max_notional and last_price:
        max_qty = min(max_qty or max_notional / last_price, max_notional / last_price)
    return {
        "qty_abs": [min_qty, max_qty if math.isfinite(max_qty) else None],
        "lot_step": step,
        "min_notional": min_notional,
        "max_notional": max_notional or None,
    }


class BinancePerpetualAdapter(BaseMarketAdapter, MarketAdapter):
    """Adapter that powers Binance USDⓈ-M perpetual strategies."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        exchange_conf = config.get("exchange") or {}
        history_conf = config.get("history") or {}
        ai_conf = config.get("ai") or {}

        self._api_key = str(exchange_conf.get("api_key") or "").strip()
        if not self._api_key:
            api_key_env = str(exchange_conf.get("api_key_env", "BINANCE_USDM_API_KEY"))
            self._api_key = os.getenv(api_key_env, "").strip()

        self._api_secret = str(exchange_conf.get("api_secret") or "").strip()
        if not self._api_secret:
            api_secret_env = str(exchange_conf.get("api_secret_env", "BINANCE_USDM_API_SECRET"))
            self._api_secret = os.getenv(api_secret_env, "").strip()

        self._passphrase = str(exchange_conf.get("password") or "").strip()
        data_kwargs = exchange_conf.get("data_exchange_kwargs") or {}

        settle_assets = exchange_conf.get("settle_assets") or ["USDT"]

        chart_conf = dict(history_conf.get("chart") or {})
        ai_chart_conf = ai_conf.get("chart") or {}
        if ai_chart_conf:
            chart_conf.update(ai_chart_conf)

        chart_enabled = bool(chart_conf.get("enabled", False))
        chart_lookback = int(chart_conf.get("lookback", 20) or 20)
        chart_dir = chart_conf.get("output_dir") or os.path.join("rockstock", "charts")
        chart_indicators_conf = chart_conf.get("indicators")
        chart_intervals_conf = chart_conf.get("intervals")

        timeframes_conf = history_conf.get("timeframes")
        if not isinstance(timeframes_conf, list):
            timeframes_conf = []

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        timeframe_entries: List[Dict[str, Any]] = []

        for idx, entry in enumerate(timeframes_conf):
            if not isinstance(entry, dict):
                continue
            tf = str(entry.get("timeframe") or "").lower()
            if not tf:
                continue
            default_history = 200 if idx == 0 else 100
            bars = _coerce_int(entry.get("bars"), default_history)
            fetch_default = max(bars, chart_lookback) if chart_enabled else bars
            fetch = _coerce_int(entry.get("fetch"), fetch_default)
            ema_entry = entry.get("ema") if isinstance(entry.get("ema"), dict) else {}
            timeframe_entries.append(
                {
                    "tf": tf,
                    "bars": max(bars, 1),
                    "fetch": fetch,
                    "ema": ema_entry,
                }
            )

        default_timeframes = [
            {"tf": "1h", "bars": 20, "fetch": 200, "ema": {"short": 20, "mid": 50, "long": 200}},
            {"tf": "4h", "bars": 10, "fetch": 100, "ema": {"short": 55, "mid": None, "long": 200}},
        ]

        (
            self._timeframes,
            self._history_bars,
            self._fetch_bars,
            self._ema_config,
        ) = self._normalize_timeframes(
            timeframe_entries,
            chart_enabled=chart_enabled,
            chart_lookback=chart_lookback,
            default_entries=default_timeframes,
        )

        self._primary_timeframe = self._timeframes[0]

        self._rsi_period = _coerce_int(history_conf.get("rsi_period"), 14)
        self._atr_period = _coerce_int(history_conf.get("atr_period"), 14)

        if chart_intervals_conf:
            requested_intervals: List[str] = []
            for interval in chart_intervals_conf:
                value = str(interval or "").strip()
                if not value:
                    continue
                requested_intervals.append(value.lower())
        else:
            requested_intervals = list(self._timeframes)
        seen_intervals: Dict[str, bool] = {}
        normalized_intervals: List[str] = []
        for interval in requested_intervals:
            interval_norm = interval.strip().lower()
            if not interval_norm or interval_norm in seen_intervals:
                continue
            normalized_intervals.append(interval_norm)
            seen_intervals[interval_norm] = True
        if not normalized_intervals and self._timeframes:
            normalized_intervals.append(self._timeframes[0])
        self._configure_chart(
            enabled=chart_enabled,
            lookback=chart_lookback,
            output_dir=chart_dir,
            intervals=normalized_intervals,
        )
        if isinstance(chart_indicators_conf, list):
            self.set_chart_indicators(chart_indicators_conf)
        elif chart_indicators_conf is None:
            self.set_chart_indicators(None)

        rate_conf = config.get("rate_limit") or {}
        min_interval = ensure_float(rate_conf.get("min_interval_seconds"), 0.25)
        retry_attempts = int(rate_conf.get("retry_attempts", 5))
        self._providers: Dict[str, BinanceProvider] = {}
        for tf in self._timeframes:
            provider = BinanceProvider(
                market="perp",
                timeframe=tf,
                limit=self._fetch_bars[tf],
                exchange_kwargs=data_kwargs,
                settle_assets=settle_assets,
                rate_limit_seconds=min_interval,
                retry_attempts=retry_attempts,
            )
            self._providers[tf] = provider

        self._exchange: Optional[ccxt_async.binanceusdm] = None  # type: ignore[assignment]
        self._markets: Dict[str, Any] = {}

        execution_conf = config.get("execution") or {}
        self._notional_per_slice = float(execution_conf.get("notional_per_slice", 5000.0))
        self._max_slices = int(execution_conf.get("max_slices", 10))
        self._slice_pause = float(execution_conf.get("slice_pause", 0.3))
        self._ignore_remaining_notional = float(execution_conf.get("ignore_remaining_notional", 100.0))

    # ------------------------------------------------------------------
    # Rounding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_account_number(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(number, 2)

    @staticmethod
    def _round_price(value: Any) -> float:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return 0.0
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
        return round(price, decimals)

    @staticmethod
    def _round_indicator_numeric(value: Optional[float], decimals: int) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(float(value), decimals)
        except (TypeError, ValueError):
            return None

    def _round_indicator_price(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return self._round_price(value)

    def _prepare_macd_details(self, closes: Sequence[float], count: int = 5) -> Dict[str, Any]:
        try:
            series = macd_series(closes, 12, 26, 9)
        except ValueError:
            return {"state": None, "macd": [], "signal": [], "hist": []}

        macd_vals = series.macd
        signal_vals = series.signal
        hist_vals = series.histogram
        if not macd_vals or not signal_vals:
            return {"state": None, "macd": [], "signal": [], "hist": []}

        macd_value = macd_vals[-1]
        signal_value = signal_vals[-1] if signal_vals else None
        hist_value = hist_vals[-1] if hist_vals else None
        state: Optional[str] = None
        if signal_value is not None and len(macd_vals) >= 2 and len(signal_vals) >= 2:
            prev_macd = macd_vals[-2]
            prev_signal = signal_vals[-2]
            if macd_value >= signal_value and prev_macd < prev_signal:
                state = "golden_cross"
            elif macd_value <= signal_value and prev_macd > prev_signal:
                state = "death_cross"
            elif hist_value is not None:
                state = "bullish_hist" if hist_value >= 0 else "bearish_hist"

        return {
            "state": state,
            "macd": self._round_price_series(macd_vals, count=count),
            "signal": self._round_price_series(signal_vals, count=count),
            "hist": self._round_price_series(hist_vals, count=count),
        }

    def _compute_indicator_snapshots(
        self,
        timeframe_data: Dict[str, Dict[str, Any]],
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, Dict[str, Any]],
        Dict[str, Optional[float]],
    ]:
        return self._build_indicator_snapshots(
            timeframe_data,
            ema_config=self._ema_config,
            rsi_period=self._rsi_period,
            atr_period=self._atr_period,
            round_price=self._round_indicator_price,
            prepare_macd=self._prepare_macd_details,
        )

    # ------------------------------------------------------------------
    # MarketAdapter interface
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        return normalize_symbol(symbol)

    async def setup(self) -> None:
        if self._exchange is not None:
            return
        params = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "password": self._passphrase or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        params = {k: v for k, v in params.items() if v is not None}
        self._exchange = ccxt_async.binanceusdm(params)
        await self._exchange.load_markets()
        self._markets = self._exchange.markets

    async def close(self) -> None:
        if self._exchange is not None:
            with contextlib.suppress(Exception):
                await self._exchange.close()
            self._exchange = None

    async def fetch_account_state(self) -> Dict[str, Any]:
        if self._exchange is None:
            await self.setup()
        assert self._exchange is not None
        try:
            balance = await self._exchange.fetch_balance(params={"type": "future"})
        except Exception as exc:  # pragma: no cover - network/service errors
            self.logger.error("账户信息获取失败: %s", exc)
            return {}

        info = balance.get("info") or {}
        available = ensure_float(info.get("availableBalance"))
        margin = ensure_float(info.get("totalInitialMargin"))
        equity = available + margin

        positions_list: List[Dict[str, Any]] = []
        positions_map: Dict[str, Dict[str, Any]] = {}
        ticker_cache: Dict[str, Any] = {}
        total_pos_notional = 0.0
        for item in info.get("positions", []):
            symbol = str(item.get("symbol") or "")
            normalized = self.normalize_symbol(symbol)
            amount = ensure_float(item.get("positionAmt"))
            if math.isclose(amount, 0.0, abs_tol=1e-8):
                continue
            entry_price = ensure_float(item.get("entryPrice"))
            if math.isclose(entry_price, 0.0, abs_tol=1e-12):
                notional = ensure_float(item.get("notional"))
                unrealized = ensure_float(item.get("unrealizedProfit"))
                if not math.isclose(amount, 0.0, abs_tol=1e-12):
                    derived_entry = (notional - unrealized) / amount
                    if math.isfinite(derived_entry) and not math.isclose(derived_entry, 0.0, abs_tol=1e-12):
                        entry_price = derived_entry
            leverage = ensure_float(item.get("leverage"))
            unrealized = ensure_float(item.get("unrealizedProfit"))

            ccxt_symbol = to_ccxt_symbol(normalized, self._markets)
            mark_price = 0.0
            if ccxt_symbol:
                ticker = ticker_cache.get(ccxt_symbol)
                if ticker is None:
                    try:
                        ticker = await self._exchange.fetch_ticker(ccxt_symbol)
                    except Exception as exc:  # pragma: no cover - network/service errors
                        self.logger.debug("Ticker fetch failed for %s: %s", ccxt_symbol, exc)
                        ticker = {}
                    ticker_cache[ccxt_symbol] = ticker
                info_block = ticker.get("info") if isinstance(ticker, dict) else None
                mark_price = ensure_float(
                    (ticker or {}).get("markPrice")
                    or (ticker or {}).get("last")
                    or (ticker or {}).get("close")
                    or (info_block or {}).get("markPrice")
                    or (info_block or {}).get("lastPrice")
                )
            if math.isclose(mark_price, 0.0, abs_tol=1e-8):
                mark_price = entry_price

            rounded_entry = self._round_price(entry_price)
            rounded_mark = self._round_price(mark_price)
            rounded_unrealized = self._round_account_number(unrealized)
            amount_abs = abs(amount)
            side = "long" if amount > 0 else "short"
            position_entry = {
                "symbol": normalized,
                "side": side,
                "amount": amount_abs,
                "entry": rounded_entry,
                "upnl": rounded_unrealized,
                "mark": rounded_mark,
            }
            positions_list.append(position_entry)
            positions_map[normalized] = position_entry
            notional_val = amount_abs * (rounded_mark or rounded_entry)
            total_pos_notional += notional_val

        account_state = {
            "net_value": self._round_account_number(equity),
            "margin_free": self._round_account_number(available),
            "margin_used": self._round_account_number(margin),
            "total_pos_notional_value": self._round_account_number(total_pos_notional),
            "positions": positions_list,
        }

        preview_positions = ", ".join(
            f"{pos['symbol']}:{pos['side']} {pos['amount']} {pos['upnl']}"
            for pos in positions_list
        )
        if not preview_positions:
            preview_positions = "none"

        self.logger.info(
            "Account snapshot fetched: net_value=%.2f margin_free=%.2f margin_used=%.2f total_notional=%.2f positions=%s",
            equity,
            available,
            margin,
            total_pos_notional,
            preview_positions,
        )
        return account_state

    async def collect_symbol_data(
        self,
        symbol: str,
        account_state: Dict[str, Any],
        *,
        position: Optional[Position],
        risk: Dict[str, Any],
        constraints: Dict[str, Any],
        use_chart: bool = True,
    ) -> Optional[SymbolData]:
        if self._exchange is None:
            await self.setup()
        assert self._exchange is not None

        ccxt_symbol = to_ccxt_symbol(symbol, self._markets)

        timeframe_data: Dict[str, Dict[str, Any]] = {}
        last_closed_ts: Optional[int] = None
        last_close = 0.0

        for tf in self._timeframes:
            provider = self._providers[tf]
            try:
                raw = await provider.get_ohlcv(ccxt_symbol)
            except Exception as exc:  # pragma: no cover - network/service errors
                self.logger.error("K线获取失败 %s@%s: %s", symbol, tf, exc)
                return None
            if not raw:
                return None
            seconds = timeframe_to_seconds(getattr(provider, "timeframe", tf)) or timeframe_to_seconds(tf) or 0
            closed, closed_ts = trim_to_closed_bars(raw, seconds)
            if not closed or seconds <= 0:
                return None
            payload_bars: List[List[Any]] = [list(bar) for bar in closed]
            if tf != self._primary_timeframe and raw:
                last_raw = list(raw[-1])
                if not payload_bars or last_raw[0] != payload_bars[-1][0]:
                    payload_bars.append(last_raw)
            fetch_cap = self._fetch_bars.get(tf)
            if fetch_cap:
                payload_bars = payload_bars[-max(int(fetch_cap), 1):]
            if tf == self._primary_timeframe:
                if closed_ts is None:
                    return None
                last_closed_ts = closed_ts
                last_close = float(closed[-1][4])
            timeframe_data[tf] = {
                "raw": raw,
                "closed": closed,
                "payload_bars": payload_bars,
                "seconds": seconds,
                "close_series": [float(item[4]) for item in payload_bars],
            }

        if last_closed_ts is None:
            return None

        live_price = last_close
        if self._exchange:
            try:
                ticker = await self._exchange.fetch_ticker(ccxt_symbol)
                live_price = ensure_float(
                    ticker.get("last")
                    or ticker.get("close")
                    or ticker.get("info", {}).get("lastPrice")
                    or ticker.get("info", {}).get("markPrice"),
                    last_close,
                )
            except Exception:
                self.logger.warning(
                    "Failed to fetch live ticker for %s, fallback to last close", symbol
                )

        self.logger.info(
            "Fetching: sym=%s last_close[%s]=%.4f last=%.4f",
            symbol,
            self._primary_timeframe,
            last_close,
            live_price,
        )

        ema_snapshot: Dict[str, List[float]] = {}
        rsi_snapshot: Dict[str, List[float]] = {}
        macd_snapshot: Dict[str, Dict[str, Any]] = {}
        atr_snapshot: Dict[str, Optional[float]] = {}

        for tf, data_map in timeframe_data.items():
            close_series = data_map["close_series"]
            ema_conf = self._ema_config.get(tf, {})
            if close_series:
                short_period = ema_conf.get("short")
                if short_period and len(close_series) >= short_period:
                    short_series = ema_series(close_series, short_period)
                    values = self._round_price_series(short_series, count=5)
                    if values:
                        ema_snapshot[f"{tf}_{short_period}"] = values

                mid_period = ema_conf.get("mid")
                if mid_period and len(close_series) >= mid_period:
                    mid_series = ema_series(close_series, mid_period)
                    values = self._round_price_series(mid_series, count=5)
                    if values:
                        ema_snapshot[f"{tf}_{mid_period}"] = values

                long_period = ema_conf.get("long")
                if long_period and len(close_series) >= long_period:
                    long_series = ema_series(close_series, long_period)
                    values = self._round_price_series(long_series, count=5)
                    if values:
                        ema_snapshot[f"{tf}_{long_period}"] = values

                rsi_series_values = [
                    value
                    for value in rsi_series(close_series, self._rsi_period)
                    if not math.isnan(value)
                ]
                rsi_values = self._round_numeric_series(rsi_series_values, count=5, decimals=2)
                if rsi_values:
                    rsi_snapshot[tf] = rsi_values

                macd_details = self._prepare_macd_details(close_series)
                if any(macd_details.values()):
                    macd_snapshot[tf] = macd_details

            payload_bars = data_map["payload_bars"]
            if len(payload_bars) >= self._atr_period:
                atr_value = atr(payload_bars, period=self._atr_period)
                atr_snapshot[tf] = self._round_indicator_price(atr_value)
            else:
                atr_snapshot[tf] = None

        market = self._markets.get(ccxt_symbol.replace("/", ":")) or self._markets.get(ccxt_symbol)
        bounds = _build_bounds(symbol, market, live_price)

        positions_map = self._positions_to_map(account_state.get("positions"))
        exchange_position = positions_map.get(self.normalize_symbol(symbol)) or {}

        payload_position: Optional[Dict[str, Any]] = None
        if position:
            payload_position = position.to_dict()
        elif exchange_position:
            payload_position = dict(exchange_position)

        payload = {
            "sym": symbol,
            "position": payload_position,
            "indicators": {
                "ema": ema_snapshot,
                "rsi": rsi_snapshot,
                "macd": macd_snapshot,
                "atr": atr_snapshot,
            },
        }

        for tf, data_map in timeframe_data.items():
            history_len = self._history_bars.get(tf, len(data_map["payload_bars"]))
            payload[f"kline_{tf}"] = [
                _format_bar(bar) for bar in data_map["payload_bars"][-history_len:]
            ]

        chart_paths: List[str] = []
        if self._chart_enabled and use_chart:
            chart_paths = await asyncio.to_thread(
                self._build_chart_images,
                symbol,
                {tf: data_map["payload_bars"] for tf, data_map in timeframe_data.items()},
            )
            if chart_paths:
                payload["chart_images"] = chart_paths

        timeframes_meta: Dict[str, int] = {}
        for tf in self._timeframes:
            data_map = timeframe_data.get(tf)
            if not data_map:
                continue
            timeframes_meta[tf] = data_map["seconds"]
        if not timeframes_meta:
            for tf, data_map in timeframe_data.items():
                seconds = ensure_float(data_map.get("seconds"), 0)
                if seconds:
                    timeframes_meta[tf] = int(seconds)

        primary_seconds: float = 0.0
        primary_tf = self._primary_timeframe
        primary_data = timeframe_data.get(primary_tf)
        if primary_data:
            primary_seconds = float(primary_data["seconds"])
        elif timeframes_meta:
            primary_seconds = float(next(iter(timeframes_meta.values())))
        if primary_seconds <= 0:
            primary_seconds = timeframe_to_seconds(primary_tf) or 3600

        return SymbolData(
            symbol=symbol,
            payload=payload,
            last_ts=last_closed_ts,
            last_price=live_price,
            meta={
                "ccxt_symbol": ccxt_symbol,
                "market": market,
                "suggested_interval_seconds": primary_seconds,
                "timeframes": timeframes_meta,
                "bounds": bounds,
                "chart_images": chart_paths,
            },
            position=position,
        )

    async def execute_decision(
        self,
        symbol: str,
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._exchange is None:
            await self.setup()
        assert self._exchange is not None

        action = str(decision.get("action") or "").upper()
        if not action:
            return None

        non_order_actions = {
            "HOLD",
            "NONE",
            "WAIT",
            "RISK_OFF",
            "ADJUST_SL",
            "ADJUST_TP",
            "TRAIL_ON",
            "TRAIL_OFF",
        }
        if action in non_order_actions:
            self.logger.info("No order required for action=%s symbol=%s", action, symbol)
            return None

        order_actions = {"BUY", "SELL", "ADD", "REDUCE", "CLOSE", "PARTIAL_TP"}
        if action not in order_actions:
            self.logger.warning("Unsupported AI action: %s", action)
            return None

        qty_raw = decision.get("amount")
        if qty_raw is None:
            self.logger.warning("AI决策缺少数量: %s", decision)
            return None
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError):
            self.logger.warning("AI数量无效: %s", decision)
            return None
        if qty <= 0:
            self.logger.warning("AI数量<=0: %s", decision)
            return None

        ccxt_symbol = data.meta.get("ccxt_symbol") or to_ccxt_symbol(symbol, self._markets)

        amount = self._quantity_to_precision(ccxt_symbol, qty)

        positions_map = self._positions_to_map(account_state.get("positions"))
        current_pos = positions_map.get(self.normalize_symbol(symbol))

        if action == "CLOSE":
            if not current_pos:
                return None
            current_amount = ensure_float(current_pos.get("amount"))
            amount = self._quantity_to_precision(ccxt_symbol, current_amount)
            if amount < current_amount:
                min_step = self._min_remaining_amount(ccxt_symbol)
                adjusted = current_amount + min_step
                amount = self._quantity_to_precision(ccxt_symbol, adjusted)
        elif action in {"REDUCE", "PARTIAL_TP"}:
            if not current_pos:
                return None
            current_amount = ensure_float(current_pos.get("amount"))
            amount = min(current_amount, amount)
            amount = self._quantity_to_precision(ccxt_symbol, amount)
            if amount <= 0 and current_amount > 0:
                amount = self._quantity_to_precision(ccxt_symbol, current_amount)
            if amount > current_amount:
                amount = current_amount

        if amount <= 0:
            self.logger.warning("数量按精度四舍五入后<=0: %s", decision)
            return None

        side = self._resolve_side(symbol, action, positions_map)
        if side is None:
            self.logger.info("未能解析交易方向: %s", decision)
            return None

        params: Dict[str, Any] = {}
        if action in {"REDUCE", "CLOSE", "PARTIAL_TP"}:
            params["reduceOnly"] = True

        requested_qty = amount
        remaining = amount
        sub_orders: List[Dict[str, Any]] = []
        total_filled = 0.0
        notional_accum = 0.0
        attempts = 0
        price_ref = ensure_float(data.last_price) or ensure_float(current_pos.get("mark") if current_pos else 0.0) or 1.0

        slice_queue = self._prepare_slice_plan(ccxt_symbol, requested_qty, price_ref)
        if not slice_queue:
            slice_queue = [self._quantity_to_precision(ccxt_symbol, requested_qty)]

        while remaining > 0 and attempts < self._max_slices and slice_queue:
            remaining_notional = remaining * max(price_ref, 1e-8)
            if remaining_notional <= self._ignore_remaining_notional:
                # self.logger.info(
                #     "Skip tiny remainder: sym=%s action=%s remaining_qty=%s notional=%s",
                #     symbol,
                #     action,
                #     remaining,
                #     remaining_notional,
                # )
                break

            target_qty = slice_queue.pop(0)
            slice_qty = min(remaining, target_qty)
            slice_qty = self._quantity_to_precision(ccxt_symbol, slice_qty)
            if slice_qty <= 0:
                continue
            attempts += 1
            try:
                order = await self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type="market",
                    side=side,
                    amount=slice_qty,
                    params=params,
                )
            except Exception as exc:  # pragma: no cover - runtime errors
                self.logger.error("下单失败-子单 %s qty=%s: %s", ccxt_symbol, slice_qty, exc)
                break

            filled = self._extract_filled(order)
            total_filled += filled
            remaining = max(0.0, remaining - filled)

            exec_price = ensure_float(
                order.get("avgPrice")
                or order.get("average")
                or order.get("average_price")
                or order.get("price")
            )
            if exec_price <= 0:
                exec_price = price_ref
            else:
                price_ref = exec_price
            notional_accum += exec_price * filled

            sub_orders.append(order)

            self.logger.info(
                "Order slice submitted: sym=%s action=%s side=%s req=%s filled=%s price=%s",
                symbol,
                action,
                side,
                slice_qty,
                filled,
                exec_price,
            )

            if filled <= 0:
                self.logger.warning(
                    "Slice returned zero fill: sym=%s action=%s req=%s", symbol, action, slice_qty
                )
                break

            if remaining > 0 and not slice_queue:
                # append remaining to ensure completion in next iteration
                slice_queue.append(self._quantity_to_precision(ccxt_symbol, remaining))

            if remaining > 0 and self._slice_pause > 0:
                await asyncio.sleep(self._slice_pause)

        remaining_notional = remaining * max(price_ref, 1e-8)
        if remaining_notional > self._ignore_remaining_notional:
            self.logger.warning(
                "Execution incomplete: sym=%s action=%s requested=%s filled=%s remaining=%s",
                symbol,
                action,
                requested_qty,
                total_filled,
                remaining,
            )

        avg_price_raw = (notional_accum / total_filled) if total_filled > 0 else None
        avg_price = self._round_price(avg_price_raw) if avg_price_raw is not None else None
        filled_qty_rounded = self._quantity_to_precision(ccxt_symbol, total_filled)

        return {
            "sym": symbol,
            "action": action,
            "side": side.upper(),
            "requested_qty": requested_qty,
            "filled_qty": filled_qty_rounded,
            "avg_price": avg_price,
            "sub_orders": sub_orders,
        }

    async def place_manual_order(
        self,
        symbol: str,
        *,
        side: str,
        amount: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if self._exchange is None:
            await self.setup()
        assert self._exchange is not None

        symbol_norm = self.normalize_symbol(symbol)
        ccxt_symbol = to_ccxt_symbol(symbol_norm, self._markets)

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            self.logger.warning("Invalid manual order amount: %s", amount)
            return None
        if amount <= 0:
            self.logger.warning("Manual order amount <= 0: %s", amount)
            return None
        amount_precise = float(self._exchange.amount_to_precision(ccxt_symbol, amount))
        if amount_precise <= 0:
            self.logger.warning("Manual order amount rounded to 0: %s", amount)
            return None

        order_type = "market"
        price_precise: Optional[float] = None
        if price is not None:
            try:
                price_precise = float(self._exchange.price_to_precision(ccxt_symbol, float(price)))
            except (TypeError, ValueError):
                self.logger.warning("Invalid manual order price: %s", price)
                return None
            if price_precise <= 0:
                self.logger.warning("Manual order price <= 0: %s", price)
                return None
            order_type = "limit"

        params: Dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True

        try:
            if order_type == "market":
                order = await self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type=order_type,
                    side=side,
                    amount=amount_precise,
                    params=params,
                )
            else:
                order = await self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type=order_type,
                    side=side,
                    amount=amount_precise,
                    price=price_precise,
                    params=params,
                )
        except Exception as exc:  # pragma: no cover - runtime errors
            self.logger.error("Manual order failed %s %s amount=%s price=%s: %s", symbol_norm, side, amount, price, exc)
            return None

        return {
            "sym": symbol_norm,
            "action": f"MANUAL_{side.upper()}",
            "qty": amount_precise,
            "side": side.upper(),
            "price": price_precise,
            "order_type": order_type,
            "order": order,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_side(self, symbol: str, action: str, positions: Dict[str, Any]) -> Optional[str]:
        action = action.upper()
        long_only = bool(self.config.get("constraints", {}).get("long_only"))
        if action in {"BUY", "ADD"}:
            return "buy"
        if action == "SELL":
            return None if long_only else "sell"
        if action in {"REDUCE", "CLOSE", "PARTIAL_TP"}:
            pos = positions.get(self.normalize_symbol(symbol))
            if not pos:
                return None
            amount_val = ensure_float(pos.get("amount"))
            side_val = str(pos.get("side") or "").lower()
            contracts = amount_val if side_val != "short" else -abs(amount_val)
            return "sell" if contracts > 0 else "buy"
        return None

    def _positions_to_map(self, positions: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if not positions:
            return result
        for pos in positions:
            sym = str(pos.get("symbol") or pos.get("sym") or "").upper().replace("/", "")
            if not sym:
                continue
            amount = ensure_float(pos.get("amount"))
            side = str(pos.get("side") or "").lower()
            entry_price = self._round_price(pos.get("entry"))
            mark_price = self._round_price(pos.get("mark"))
            unrealized = self._round_account_number(pos.get("upnl"))
            result[sym] = {
                "symbol": sym,
                "side": "long" if side != "short" else "short",
                "amount": abs(amount),
                "entry": entry_price,
                "upnl": unrealized,
                "mark": mark_price,
            }
        return result

    def _min_remaining_amount(self, ccxt_symbol: str) -> float:
        market = self._markets.get(ccxt_symbol) or {}
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        min_qty = ensure_float(amount_limits.get("min"))
        step = ensure_float(amount_limits.get("step")) or market.get("contractSize") or 0.0
        return max(min_qty, step)

    def _quantity_to_precision(self, ccxt_symbol: str, qty: float) -> float:
        try:
            precise = float(self._exchange.amount_to_precision(ccxt_symbol, qty))
        except Exception:
            precise = qty
        return precise

    def _prepare_slice_plan(self, ccxt_symbol: str, amount: float, price_ref: float) -> List[float]:
        if amount <= 0:
            return []

        precise_amount = self._quantity_to_precision(ccxt_symbol, amount)
        if precise_amount <= 0:
            return []

        min_qty = self._min_remaining_amount(ccxt_symbol)
        if precise_amount <= min_qty:
            return [precise_amount]

        max_slices = max(1, self._max_slices)
        price = max(price_ref, 1e-8)
        target_slice = precise_amount
        if self._notional_per_slice > 0:
            target_slice = self._notional_per_slice / price
        target_slice = max(min_qty, target_slice)
        target_slice = self._quantity_to_precision(ccxt_symbol, target_slice)
        if target_slice <= 0:
            target_slice = min_qty

        # For small totals or when slice size is large relative to amount, keep it single order
        if max_slices <= 1 or precise_amount <= target_slice * 1.5:
            return [precise_amount]

        plan: List[float] = []
        remaining = precise_amount
        while remaining > target_slice and len(plan) < max_slices - 1:
            qty = min(target_slice, remaining)
            qty = self._quantity_to_precision(ccxt_symbol, qty)
            if qty <= 0:
                break
            plan.append(qty)
            remaining = max(0.0, remaining - qty)
            if remaining <= min_qty:
                break

        # push the final remainder as one chunk
        remainder = self._quantity_to_precision(ccxt_symbol, remaining)
        if remainder > 0:
            plan.append(remainder)

        # Merge tiny tail with previous slice if below minimum
        if len(plan) >= 2 and plan[-1] < min_qty:
            plan[-2] = self._quantity_to_precision(ccxt_symbol, plan[-2] + plan[-1])
            plan.pop()

        # As a safeguard ensure at least one slice exists
        if not plan:
            plan = [precise_amount]

        total = sum(plan)
        if total > precise_amount and plan:
            adjusted_last = max(precise_amount - sum(plan[:-1]), 0.0)
            plan[-1] = self._quantity_to_precision(ccxt_symbol, max(adjusted_last, min_qty))

        return [qty for qty in plan if qty > 0]

    @staticmethod
    def _extract_filled(order: Dict[str, Any]) -> float:
        return ensure_float(
            order.get("filled")
            or order.get("executedQty")
            or order.get("executed_quantity")
            or order.get("qty")
            or 0.0
        )

    def _refresh_price(self, _ccxt_symbol: str, fallback: float) -> float:
        return fallback


__all__ = ["BinancePerpetualAdapter"]
