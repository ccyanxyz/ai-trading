"""Market adapter for Tiger Brokers equities used by the AI scanner."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from adapters.tiger import TigerAdapter, BarPeriod
from tigeropen.common.exceptions import ApiException

from ..core.interfaces import MarketAdapter, SymbolData
from ..utils import ensure_float, timeframe_to_seconds
from .base import BaseMarketAdapter


def _normalize_symbol(text: str) -> str:
    return str(text).upper().replace("/", "").replace(":", "")


class TigerEquityAdapter(BaseMarketAdapter, MarketAdapter):
    """Fetches US equities OHLCV bars via Tiger for downstream AI consumers."""

    DEFAULT_TIMEFRAMES: Sequence[Dict[str, Any]] = (
        {"tf": "1d", "bars": 260, "fetch": 320},
        {"tf": "1w", "bars": 200, "fetch": 260},
    )

    PERIOD_ALIASES = {
        "1d": getattr(BarPeriod, "DAY", "DAY"),
        "1w": getattr(BarPeriod, "WEEK", "WEEK"),
        "1h": getattr(BarPeriod, "HOUR", "HOUR"),
        "1m": getattr(BarPeriod, "MINUTE", "MINUTE"),
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        exchange_conf = config.get("exchange") or {}
        self._tiger_conf = exchange_conf.get("tiger") or {}

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
            default_entries=self.DEFAULT_TIMEFRAMES,
        )
        if not self._timeframes:
            self._timeframes = ["1d"]
        self._primary_timeframe = self._timeframes[0]
        self._timeframe_seconds: Dict[str, int] = {
            tf: timeframe_to_seconds(tf) or 0 for tf in self._timeframes
        }

        self._tiger: Optional[TigerAdapter] = None
        self._rate_lock: asyncio.Lock = asyncio.Lock()
        self._last_request_ts: float = 0.0
        self._min_interval: float = 1.1  # Tiger limit: <=60 calls/min → spacing >1s

    # ------------------------------------------------------------------
    # MarketAdapter interface
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        return _normalize_symbol(symbol)

    async def setup(self) -> None:
        if self._tiger is None:
            self._tiger = TigerAdapter(self._tiger_conf)

    async def close(self) -> None:
        self._tiger = None

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
        await self.setup()
        if self._tiger is None:
            return None

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

        if last_ts_ms == 0:
            for rows in timeframe_payload.values():
                if rows:
                    base_row = rows[-1]
                    last_close = base_row[3]
                    last_ts_ms = int(base_row[5]) * 1000
                    break

        payload: Dict[str, Any] = {"sym": symbol_norm}
        for tf, rows in timeframe_payload.items():
            payload[f"kline_{tf}"] = rows

        timeframes_meta = {
            tf: self._timeframe_seconds.get(tf, timeframe_to_seconds(tf) or 0)
            for tf in timeframe_payload.keys()
        }

        primary_seconds = timeframes_meta.get(self._primary_timeframe) or 0

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
        raise NotImplementedError("TigerEquityAdapter does not support trade execution")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_timeframe_rows(self, symbol: str, timeframe: str) -> List[List[float]]:
        if self._tiger is None:
            return []
        period = self.PERIOD_ALIASES.get(timeframe, getattr(BarPeriod, "DAY", "DAY"))
        limit = max(self._fetch_bars.get(timeframe, 300), 1)
        bars: Any = None
        for attempt in range(5):
            await self._respect_rate_limit()
            try:
                bars = await self._tiger.get_stock_bars(symbol, period=period, limit=limit)
                break
            except ApiException as exc:
                if "rate limit" in str(exc).lower():
                    wait = min(5, 1 + attempt)
                    self.logger.warning(
                        "Tiger rate limit hit for %s %s (attempt %d). Sleeping %.1fs",
                        symbol,
                        timeframe,
                        attempt + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                self.logger.exception("Tiger API error for %s %s", symbol, timeframe)
                return []
            except Exception:
                self.logger.exception("Failed to fetch stock bars for %s %s", symbol, timeframe)
                return []
        if bars is None:
            self.logger.error("Exceeded retries fetching stock bars for %s %s", symbol, timeframe)
            return []
        records = self._bars_to_records(bars)
        if not records:
            return []
        return self._records_to_rows(records)

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()

    @staticmethod
    def _bars_to_records(bars: Any) -> List[Dict[str, Any]]:
        if bars is None:
            return []
        if isinstance(bars, pd.DataFrame):
            return bars.to_dict(orient="records")
        if isinstance(bars, dict):
            for key in ("items", "data", "bars"):
                if key in bars and isinstance(bars[key], (list, tuple)):
                    return list(bars[key])
            return [bars]
        if isinstance(bars, (list, tuple)):
            return [record for record in bars if record is not None]
        try:
            return list(bars)
        except Exception:
            return []

    def _records_to_rows(self, records: Iterable[Dict[str, Any]]) -> List[List[float]]:
        df = pd.DataFrame(list(records))
        if df.empty:
            return []
        df.columns = [str(col).lower() for col in df.columns]

        time_col = None
        for candidate in ("time", "timestamp", "end_time", "begintime", "datetime"):
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col is None:
            return []

        try:
            ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        except Exception:
            ts = pd.Series([pd.NaT] * len(df))
        try:
            ts_int = (ts.astype("int64") // 10**9)
        except Exception:
            ts_int = pd.Series([pd.NA] * len(df))
        df = df.assign(_ts=ts_int)
        df = df[df["_ts"].notna()]
        df = df.drop_duplicates(subset=["_ts"], keep="last")

        column_map = {}
        for name in ("open", "high", "low", "close", "volume"):
            if name in df.columns:
                column_map[name] = name
            else:
                alt = name[0].upper() + name[1:]
                if alt.lower() in df.columns:
                    column_map[name] = alt.lower()
                elif alt in df.columns:
                    df[alt.lower()] = df[alt]
                    column_map[name] = alt.lower()
        required = {"open", "high", "low", "close"}
        if not required.issubset(column_map.keys()):
            return []

        rows: List[List[float]] = []
        df = df.sort_values("_ts")
        for _, row in df.iterrows():
            ts_value = ensure_float(row.get("_ts"))
            if ts_value <= 0:
                continue
            open_price = ensure_float(row.get(column_map["open"]))
            high = ensure_float(row.get(column_map["high"]))
            low = ensure_float(row.get(column_map["low"]))
            close = ensure_float(row.get(column_map["close"]))
            volume = ensure_float(row.get(column_map.get("volume", "volume")), 0.0)
            rows.append([open_price, high, low, close, volume, int(ts_value)])
        return rows


__all__ = ["TigerEquityAdapter"]
