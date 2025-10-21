"""Shared helpers for concrete market adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
from typing import Callable

from ..utils import ensure_float, round_numeric_series, round_price_series
from ..services.chart import ChartService, ChartConfig, klines_to_dataframe
from signals.indicators import ema_series, rsi_series, atr, macd_series


class BaseMarketAdapter:
    """Provides chart management and rounding helpers for market adapters."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self._chart_enabled: bool = False
        self._chart_lookback: int = 0
        self._chart_dir: Path = Path("charts")
        self._chart_intervals: Sequence[str] = ()
        self._chart_last_ts: Dict[Tuple[str, str], int] = {}
        self._chart_indicators: Optional[List[str]] = None
        self._chart_lock: Lock = Lock()

    def _configure_chart(
        self,
        *,
        enabled: bool,
        lookback: int,
        output_dir: str,
        intervals: Sequence[str],
    ) -> None:
        self._chart_enabled = bool(enabled)
        self._chart_lookback = int(lookback)
        self._chart_dir = Path(output_dir)
        if self._chart_enabled:
            self._chart_dir.mkdir(parents=True, exist_ok=True)
        self._chart_intervals = tuple(intervals)
        self._chart_last_ts = {}

    def set_chart_indicators(self, indicators: Optional[Sequence[str]]) -> None:
        if indicators is None:
            self._chart_indicators = None
        else:
            parsed = [str(item).lower() for item in indicators if item]
            self._chart_indicators = parsed

    @staticmethod
    def _as_optional_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_timeframes(
        self,
        entries: Sequence[Dict[str, Any]],
        *,
        chart_enabled: bool,
        chart_lookback: int,
        default_entries: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, Dict[str, Optional[int]]]]:
        normalized: List[Tuple[str, int, int, Dict[str, Optional[int]]]] = []
        seen: Dict[str, bool] = {}

        def _coerce_entry(entry: Dict[str, Any]) -> Optional[Tuple[str, int, int, Dict[str, Optional[int]]]]:
            tf_text = str(entry.get("tf") or entry.get("timeframe") or "").strip().lower()
            if not tf_text or tf_text in seen:
                return None
            bars = int(entry.get("bars") or 0)
            if bars <= 0:
                bars = 1
            fetch_raw = int(entry.get("fetch") or 0)
            fetch = fetch_raw if fetch_raw > 0 else bars
            if chart_enabled:
                fetch = max(fetch, chart_lookback)
            fetch = max(fetch, bars)
            ema_entry = entry.get("ema") if isinstance(entry.get("ema"), dict) else {}
            ema_config = {
                "short": self._as_optional_int(ema_entry.get("short")),
                "mid": self._as_optional_int(ema_entry.get("mid")),
                "long": self._as_optional_int(ema_entry.get("long")),
            }
            seen[tf_text] = True
            return tf_text, bars, fetch, ema_config

        for entry in entries:
            coerced = _coerce_entry(entry)
            if coerced:
                normalized.append(coerced)

        if not normalized:
            seen.clear()
            for entry in default_entries:
                coerced = _coerce_entry(entry)
                if coerced:
                    normalized.append(coerced)

        timeframes = [item[0] for item in normalized]
        history_bars = {item[0]: item[1] for item in normalized}
        fetch_bars = {item[0]: item[2] for item in normalized}
        ema_config = {item[0]: item[3] for item in normalized}
        return timeframes, history_bars, fetch_bars, ema_config

    def _build_chart_images(
        self,
        symbol: str,
        kline_map: Dict[str, Sequence[Sequence[Any]]],
    ) -> List[str]:
        if not self._chart_enabled:
            return []
        with self._chart_lock:
            return self._build_chart_images_locked(symbol, kline_map)

    def _build_chart_images_locked(
        self,
        symbol: str,
        kline_map: Dict[str, Sequence[Sequence[Any]]],
    ) -> List[str]:
        paths: List[str] = []
        symbol_upper = symbol.upper()
        for interval in self._chart_intervals:
            klines = kline_map.get(interval)
            if not klines:
                continue
            try:
                last_ts = int(ensure_float(klines[-1][0]))
            except (TypeError, ValueError):
                continue
            key = (symbol_upper, interval)
            chart_path = self._chart_dir / f"{symbol_upper}_{interval}.png"
            if self._chart_last_ts.get(key) == last_ts and chart_path.exists():
                paths.append(str(chart_path))
                continue
            df = klines_to_dataframe(klines, limit=self._chart_lookback)
            if df.empty:
                continue
            try:
                ChartService.render(
                    df,
                    ChartConfig(
                        symbol=symbol_upper,
                        interval=interval,
                        output_path=chart_path,
                        indicators=self._chart_indicators,
                    ),
                )
            except Exception:
                self.logger.exception("Failed to render chart for %s %s", symbol_upper, interval)
                continue
            self._chart_last_ts[key] = last_ts
            paths.append(str(chart_path))
        return paths

    def _round_price_series(
        self,
        values: Iterable[Any],
        *,
        count: int = 5,
    ) -> List[float]:
        return round_price_series(values, count=count)

    def _round_numeric_series(
        self,
        values: Iterable[Any],
        *,
        count: int,
        decimals: int,
    ) -> List[float]:
        return round_numeric_series(values, count=count, decimals=decimals)

    def _build_indicator_snapshots(
        self,
        timeframe_data: Dict[str, Dict[str, Any]],
        *,
        ema_config: Dict[str, Dict[str, Optional[int]]],
        rsi_period: int,
        atr_period: int,
        round_price: Callable[[Optional[float]], Optional[float]],
        prepare_macd: Optional[Callable[[Sequence[float]], Dict[str, Any]]] = None,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        Dict[str, Dict[str, Any]],
        Dict[str, Optional[float]],
    ]:
        ema_snapshot: Dict[str, List[float]] = {}
        rsi_snapshot: Dict[str, List[float]] = {}
        macd_snapshot: Dict[str, Dict[str, Any]] = {}
        atr_snapshot: Dict[str, Optional[float]] = {}

        for tf, data_map in timeframe_data.items():
            close_series = data_map.get("close_series") or []
            ema_conf = ema_config.get(tf, {})
            if close_series:
                short_period = ema_conf.get("short")
                if short_period and len(close_series) >= short_period:
                    ema_values = ema_series(close_series, short_period)
                    rounded = self._round_price_series(ema_values, count=5)
                    if rounded:
                        ema_snapshot[f"{tf}_{short_period}"] = rounded

                mid_period = ema_conf.get("mid")
                if mid_period and len(close_series) >= mid_period:
                    ema_values = ema_series(close_series, mid_period)
                    rounded = self._round_price_series(ema_values, count=5)
                    if rounded:
                        ema_snapshot[f"{tf}_{mid_period}"] = rounded

                long_period = ema_conf.get("long")
                if long_period and len(close_series) >= long_period:
                    ema_values = ema_series(close_series, long_period)
                    rounded = self._round_price_series(ema_values, count=5)
                    if rounded:
                        ema_snapshot[f"{tf}_{long_period}"] = rounded

                rsi_series_values = [
                    value
                    for value in rsi_series(close_series, rsi_period)
                    if not math.isnan(value)
                ]
                rsi_values = self._round_numeric_series(rsi_series_values, count=5, decimals=2)
                if rsi_values:
                    rsi_snapshot[tf] = rsi_values

                if prepare_macd:
                    macd_details = prepare_macd(close_series)
                else:
                    macd_details = self._prepare_macd_details_default(close_series)
                if macd_details and any(macd_details.values()):
                    macd_snapshot[tf] = macd_details

            payload_bars = data_map.get("payload_bars") or []
            if len(payload_bars) >= atr_period:
                atr_value = atr(payload_bars, period=atr_period)
                atr_snapshot[tf] = round_price(atr_value)
            else:
                atr_snapshot[tf] = None

        return ema_snapshot, rsi_snapshot, macd_snapshot, atr_snapshot

    def _prepare_macd_details_default(self, close_series: Sequence[float]) -> Dict[str, Any]:
        try:
            macd_result = macd_series(close_series)
        except ValueError:
            return {"macd": [], "signal": [], "hist": []}
        macd_line = getattr(macd_result, "macd", None) or []
        signal_line = getattr(macd_result, "signal", None) or []
        hist_line = getattr(macd_result, "histogram", None) or getattr(macd_result, "hist", None) or []
        return {
            "macd": self._round_price_series(macd_line, count=5) if macd_line else [],
            "signal": self._round_price_series(signal_line, count=5) if signal_line else [],
            "hist": self._round_price_series(hist_line, count=5) if hist_line else [],
        }


__all__ = ["BaseMarketAdapter"]
