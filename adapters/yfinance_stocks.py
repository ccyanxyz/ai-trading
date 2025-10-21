"""Market adapter for equities using yfinance as the data source."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..core.interfaces import MarketAdapter, SymbolData
from ..utils import ensure_float, timeframe_to_seconds
from .base import BaseMarketAdapter

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _normalize_symbol(text: str) -> str:
    return text.upper().replace("/", "").replace(":", "")


class YFinanceEquityAdapter(BaseMarketAdapter, MarketAdapter):
    """Lightweight adapter that fetches OHLCV data via yfinance."""

    INTERVAL_MAP = {
        "1d": "1d",
        "1w": "1wk",
        "1wk": "1wk",
        "1mo": "1mo",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if yf is None:
            raise RuntimeError("yfinance is required for YFinanceEquityAdapter")

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
            default_entries=[{"tf": "1d", "bars": 260, "fetch": 320}],
        )
        if not self._timeframes:
            self._timeframes = ["1d"]
        self._primary_timeframe = self._timeframes[0]
        self._timeframe_seconds: Dict[str, int] = {
            tf: timeframe_to_seconds(tf) or 0 for tf in self._timeframes
        }

        rate_conf = config.get("rate_limit") or {}
        min_interval = ensure_float(rate_conf.get("min_interval_seconds"), 0.6)
        self._min_interval = max(min_interval, 0.1)
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

    async def execute_decision(  # pragma: no cover - not used by scanner
        self,
        symbol: str,
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("YFinanceEquityAdapter does not support execution")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_timeframe_rows(self, symbol: str, timeframe: str) -> List[List[float]]:
        interval = self.INTERVAL_MAP.get(timeframe, timeframe)
        if interval not in {"1d", "1wk", "1mo"}:
            interval = "1d"
        limit = max(self._fetch_bars.get(timeframe, 240), 60)

        await self._respect_rate_limit()
        try:
            df = await asyncio.to_thread(
                self._download_history,
                symbol,
                interval,
                limit,
            )
        except Exception:  # pragma: no cover - network errors
            self.logger.exception("Failed to fetch yfinance data for %s %s", symbol, timeframe)
            return []

        if df is None or df.empty:
            return []

        df = df.tail(limit)
        rows: List[List[float]] = []
        for ts, row in df.iterrows():
            try:
                open_price = ensure_float(row["Open"])
                high = ensure_float(row["High"])
                low = ensure_float(row["Low"])
                close = ensure_float(row["Close"])
                volume = ensure_float(row.get("Volume", 0.0))
                ts_int = int(pd.Timestamp(ts).timestamp())
            except Exception:
                continue
            rows.append([open_price, high, low, close, volume, ts_int])
        return rows

    def _download_history(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        ticker = yf.Ticker(symbol)
        seconds = timeframe_to_seconds(self._primary_timeframe) or 86400
        days_per_bar = max(1, int(seconds / 86400))
        lookback_days = max(limit * days_per_bar * 2, 30)
        start = datetime.utcnow() - timedelta(days=lookback_days)
        try:
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=None,
                interval=interval,
                actions=False,
                auto_adjust=False,
                back_adjust=False,
                prepost=False,
            )
        except Exception:
            raise
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()


__all__ = ["YFinanceEquityAdapter"]
