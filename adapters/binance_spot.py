"""Market adapter for Binance spot markets used by the AI scanner."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data.binance import BinanceProvider

from ..core.interfaces import MarketAdapter, SymbolData
from ..utils import ensure_float, timeframe_to_seconds
from .base import BaseMarketAdapter


def _normalize_symbol(text: str) -> str:
    value = text.upper().replace(":", "").replace(" ", "")
    if "/" not in value:
        if value.endswith("USDT"):
            value = f"{value[:-4]}/USDT"
        elif value.endswith("USDC"):
            value = f"{value[:-4]}/USDC"
    return value


class BinanceSpotAdapter(BaseMarketAdapter, MarketAdapter):
    """Reads spot OHLCV data through ccxt/binance with light rate limiting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        history_conf = config.get("history") or {}
        timeframes_config = history_conf.get("timeframes") or []

        (
            self._timeframes,
            self._history_bars,
            self._fetch_bars,
            _,
        ) = self._normalize_timeframes(
            timeframes_config,
            chart_enabled=False,
            chart_lookback=0,
            default_entries=[{"tf": "1d", "bars": 180, "fetch": 240}],
        )
        if not self._timeframes:
            self._timeframes = ["1d"]
        self._primary_timeframe = self._timeframes[0]
        self._timeframe_seconds: Dict[str, int] = {
            tf: timeframe_to_seconds(tf) or 0 for tf in self._timeframes
        }

        exchange_conf = config.get("exchange") or {}
        settle_assets = exchange_conf.get("settle_assets") or ["USDT"]
        exclude_bases = exchange_conf.get("exclude_bases") or []
        provider_kwargs = exchange_conf.get("exchange_kwargs") or {}

        self._providers: Dict[str, BinanceProvider] = {}
        for tf in self._timeframes:
            provider = BinanceProvider(
                market="spot",
                timeframe=tf,
                limit=self._fetch_bars.get(tf, 240),
                exchange_kwargs=provider_kwargs,
                settle_assets=settle_assets,
                exclude_bases=exclude_bases,
            )
            self._providers[tf] = provider

        rate_conf = config.get("rate_limit") or {}
        min_interval = ensure_float(rate_conf.get("min_interval_seconds"), 1.0)
        self._min_interval = max(min_interval, 0.2)
        self._rate_lock: asyncio.Lock = asyncio.Lock()
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # MarketAdapter interface
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        return _normalize_symbol(symbol)

    async def setup(self) -> None:
        return

    async def close(self) -> None:
        return

    async def fetch_account_state(self) -> Dict[str, Any]:  # pragma: no cover - scanner only
        return {}

    async def collect_symbol_data(
        self,
        symbol: str,
        account_state: Dict[str, Any],
        *,
        position: Any = None,
        risk: Dict[str, Any],
        constraints: Dict[str, Any],
        use_chart: bool = True,
    ) -> Optional[SymbolData]:
        symbol_norm = self.normalize_symbol(symbol)

        timeframe_payload: Dict[str, List[List[float]]] = {}
        last_close = 0.0
        last_ts_ms = 0

        for tf in self._timeframes:
            rows = await self._fetch_timeframe_rows(symbol_norm, tf)
            if not rows:
                continue
            history_len = self._history_bars.get(tf, len(rows))
            timeframe_payload[tf] = rows[-history_len:]
            if tf == self._primary_timeframe:
                base_row = timeframe_payload[tf][-1]
                last_close = base_row[3]
                last_ts_ms = int(base_row[5]) * 1000

        if not timeframe_payload:
            return None

        payload: Dict[str, Any] = {"sym": symbol_norm}
        for tf, rows in timeframe_payload.items():
            payload[f"kline_{tf}"] = rows

        timeframes_meta = {
            tf: self._timeframe_seconds.get(tf, timeframe_to_seconds(tf) or 0)
            for tf in timeframe_payload.keys()
        }

        primary_seconds = timeframes_meta.get(self._primary_timeframe, 86400)

        return SymbolData(
            symbol=symbol_norm,
            payload=payload,
            last_ts=last_ts_ms,
            last_price=last_close,
            meta={
                "timeframes": timeframes_meta,
                "suggested_interval_seconds": primary_seconds,
            },
            position=None,
        )

    async def execute_decision(  # pragma: no cover - not used for scanner
        self,
        symbol: str,
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("BinanceSpotAdapter does not support execution")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_timeframe_rows(self, symbol: str, timeframe: str) -> List[List[float]]:
        provider = self._providers.get(timeframe)
        if not provider:
            return []
        await self._respect_rate_limit()
        try:
            ohlcv = await provider.get_ohlcv(symbol)
        except Exception:  # pragma: no cover - ccxt errors
            self.logger.exception("Failed to fetch Binance spot bars for %s %s", symbol, timeframe)
            return []
        rows: List[List[float]] = []
        for entry in ohlcv:
            if len(entry) < 6:
                continue
            ts_ms = ensure_float(entry[0])
            open_price = ensure_float(entry[1])
            high = ensure_float(entry[2])
            low = ensure_float(entry[3])
            close = ensure_float(entry[4])
            volume = ensure_float(entry[5])
            ts_int = int(ts_ms)
            rows.append([open_price, high, low, close, volume, ts_int // 1000])
        return rows

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()


__all__ = ["BinanceSpotAdapter"]
