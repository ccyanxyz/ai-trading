"""Market adapter implementation for Tiger Brokers futures."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from adapters.tiger import TigerAdapter, BarPeriod, SecurityType
from signals.indicators import atr, ema_series, macd_series, rsi_series

from ..core.interfaces import MarketAdapter, SymbolData
from ..core.position import Position
from ..utils import ensure_float, round_float, round_price, timeframe_to_seconds, trim_to_closed_bars
from .base import BaseMarketAdapter


# ---------------------------------------------------------------------------
class TigerFuturesAdapter(BaseMarketAdapter, MarketAdapter):
    """Adapter bridging Tiger Brokers futures data into the AI engine."""

    DEFAULT_SYMBOLS = ("GCMAIN", "SIMAIN", "HGMAIN")

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        exchange_conf = config.get("exchange") or {}
        history_conf = config.get("history") or {}
        ai_conf = config.get("ai") or {}
        futures_conf = config.get("futures") or {}

        self._tiger_conf = exchange_conf.get("tiger") or {}

        risk_conf = config.get("risk") or {}
        self._max_amount_per_symbol = ensure_float(risk_conf.get("max_amount_per_symbol"), 0.0)
        if self._max_amount_per_symbol < 0:
            self._max_amount_per_symbol = 0.0
        self._margin_free_floor_pct = ensure_float(risk_conf.get("margin_free_floor_pct"), 0.0)
        self._margin_free_floor_abs = ensure_float(risk_conf.get("margin_free_floor_abs"), 0.0)
        self._margin_buffer_pct = ensure_float(risk_conf.get("margin_buffer_pct"), 0.0)
        self._tiger: Optional[TigerAdapter] = None
        constraints_conf = config.get("constraints") or {}
        self._long_only = bool(constraints_conf.get("long_only", False))

        chart_conf = dict(history_conf.get("chart") or {})
        ai_chart_conf = ai_conf.get("chart") or {}
        if ai_chart_conf:
            chart_conf.update(ai_chart_conf)

        chart_enabled = bool(chart_conf.get("enabled", False))
        chart_lookback = int(chart_conf.get("lookback", 120) or 120)
        chart_dir = chart_conf.get("output_dir") or "charts"
        chart_indicators_conf = chart_conf.get("indicators")
        chart_intervals_conf = chart_conf.get("intervals")

        timeframes_conf_raw = history_conf.get("timeframes")
        timeframes_conf: List[Dict[str, Any]] = []
        if isinstance(timeframes_conf_raw, list):
            timeframes_conf = list(timeframes_conf_raw)
        else:
            # Backward compatibility with legacy per-timeframe keys
            if any(
                key in history_conf
                for key in ("bars_1h", "fetch_bars_1h", "ema_1h", "ema_1h_mid", "ema_1h_long")
            ):
                timeframes_conf.append(
                    {
                        "timeframe": "1h",
                        "bars": history_conf.get("bars_1h"),
                        "fetch": history_conf.get("fetch_bars_1h"),
                        "ema": {
                            "short": history_conf.get("ema_1h"),
                            "mid": history_conf.get("ema_1h_mid"),
                            "long": history_conf.get("ema_1h_long"),
                        },
                    }
                )
            if any(key in history_conf for key in ("bars_4h", "fetch_bars_4h", "ema_4h", "ema_4h_long")):
                timeframes_conf.append(
                    {
                        "timeframe": "4h",
                        "bars": history_conf.get("bars_4h"),
                        "fetch": history_conf.get("fetch_bars_4h"),
                        "ema": {
                            "short": history_conf.get("ema_4h"),
                            "mid": history_conf.get("ema_4h_mid"),
                            "long": history_conf.get("ema_4h_long"),
                        },
                    }
                )

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _coerce_optional_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        tf_entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(timeframes_conf):
            if not isinstance(entry, dict):
                continue
            tf_text = str(entry.get("timeframe") or "").strip().lower()
            if not tf_text:
                continue
            seconds = timeframe_to_seconds(tf_text)
            if not seconds:
                self.logger.warning("Unsupported timeframe %s in history config; skipping", tf_text)
                continue
            default_bars = 200 if idx == 0 else 100
            bars = _coerce_int(entry.get("bars"), default_bars)
            fetch_default = max(bars, chart_lookback) if chart_enabled else bars
            fetch = _coerce_int(entry.get("fetch"), fetch_default)
            ema_entry = entry.get("ema") if isinstance(entry.get("ema"), dict) else {}
            tf_entries.append(
                {
                    "tf": tf_text,
                    "seconds": int(seconds),
                    "bars": max(bars, 1),
                    "fetch": max(fetch, 1),
                    "ema": {
                        "short": _coerce_optional_int(ema_entry.get("short")),
                        "mid": _coerce_optional_int(ema_entry.get("mid")),
                        "long": _coerce_optional_int(ema_entry.get("long")),
                    },
                }
            )

        if not tf_entries:
            tf_entries = [
                {
                    "tf": "1h",
                    "seconds": timeframe_to_seconds("1h") or 3600,
                    "bars": 20,
                    "fetch": 240,
                    "ema": {"short": 20, "mid": 60, "long": 120},
                },
                {
                    "tf": "4h",
                    "seconds": timeframe_to_seconds("4h") or 14400,
                    "bars": 15,
                    "fetch": 180,
                    "ema": {"short": 20, "mid": None, "long": 60},
                },
            ]

        tf_entries.sort(key=lambda item: (item["seconds"], item["tf"]))

        rsi_period = _coerce_int(
            history_conf.get("rsi_period") if history_conf.get("rsi_period") is not None else history_conf.get("rsi"),
            14,
        )
        atr_period = _coerce_int(history_conf.get("atr_period"), 14)

        normalized_entries: List[Dict[str, Any]] = []
        for item in tf_entries:
            ema_conf = item["ema"]
            fetch = item["fetch"]
            ema_periods = [period for period in ema_conf.values() if isinstance(period, int) and period > 0]
            if ema_periods:
                fetch = max(fetch, max(ema_periods))
            fetch = max(fetch, rsi_period, atr_period)
            if chart_enabled:
                fetch = max(fetch, chart_lookback)
            normalized_entries.append(
                {
                    "tf": item["tf"],
                    "bars": item["bars"],
                    "fetch": fetch,
                    "ema": ema_conf,
                    "seconds": item["seconds"],
                }
            )

        default_timeframes = [
            {
                "tf": "1h",
                "bars": 20,
                "fetch": max(240, rsi_period, atr_period, chart_lookback if chart_enabled else 0),
                "ema": {"short": 20, "mid": 60, "long": 120},
                "seconds": timeframe_to_seconds("1h") or 3600,
            },
            {
                "tf": "4h",
                "bars": 15,
                "fetch": max(180, rsi_period, atr_period, chart_lookback if chart_enabled else 0),
                "ema": {"short": 20, "mid": None, "long": 60},
                "seconds": timeframe_to_seconds("4h") or 14400,
            },
        ]

        (
            self._timeframes,
            self._history_bars,
            self._fetch_bars,
            self._ema_config,
        ) = self._normalize_timeframes(
            normalized_entries,
            chart_enabled=chart_enabled,
            chart_lookback=chart_lookback,
            default_entries=default_timeframes,
        )

        seconds_map: Dict[str, int] = {
            item["tf"]: int(item.get("seconds") or 0) for item in normalized_entries if item.get("tf")
        }
        if not seconds_map:
            seconds_map = {
                entry["tf"]: int(entry.get("seconds") or timeframe_to_seconds(entry["tf"]) or 0)
                for entry in default_timeframes
            }
        self._timeframe_seconds: Dict[str, int] = {}
        for tf in self._timeframes:
            seconds = seconds_map.get(tf)
            if not seconds or seconds <= 0:
                seconds = timeframe_to_seconds(tf) or 0
            self._timeframe_seconds[tf] = seconds

        self._primary_timeframe = self._timeframes[0]

        self._rsi_period = rsi_period
        self._atr_period = atr_period

        if chart_intervals_conf:
            intervals: List[str] = []
            for value in chart_intervals_conf:
                tf_value = str(value or "").strip().lower()
                if not tf_value or tf_value in intervals:
                    continue
                intervals.append(tf_value)
        else:
            intervals = list(self._timeframes)
        if not intervals:
            intervals = [self._primary_timeframe]
        self._configure_chart(
            enabled=chart_enabled,
            lookback=chart_lookback,
            output_dir=chart_dir,
            intervals=intervals,
        )
        if isinstance(chart_indicators_conf, list):
            self.set_chart_indicators(chart_indicators_conf)
        elif chart_indicators_conf is None:
            self.set_chart_indicators(None)
        self._market_status_cache: Dict[str, Tuple[float, bool]] = {}
        futures_market_ttl = futures_conf.get("market_status_ttl_seconds")
        self._market_status_ttl = float(futures_market_ttl if futures_market_ttl is not None else 60.0)

        self._symbol_meta, self._alias_lookup = self._build_symbol_metadata(futures_conf)
        self._contract_mapping_main_to_active: Dict[str, str] = {}
        self._contract_mapping_active_to_main: Dict[str, str] = {}
        self._load_contract_mapping(futures_conf)
        self._prefix_to_template: Dict[str, str] = self._build_prefix_map()
        self._metadata_fetch_failures: Set[str] = set()
        if not self._symbol_meta:
            for alias in self.DEFAULT_SYMBOLS:
                alias_u = alias.upper()
                self._symbol_meta[alias_u] = {
                    "data_symbol": alias_u,
                    "trade_symbol": alias_u,
                    "sec_type": "FUT",
                    "exchange": "NYMEX" if alias_u.startswith("G") else "COMEX",
                    "currency": "USD",
                    "lot_size": 1.0,
                    "min_qty": 1.0,
                }
                self._alias_lookup[alias_u] = alias_u

        asset_segments_conf = futures_conf.get("asset_segments")
        if asset_segments_conf is None:
            default_segments = ("FUT", "FUTURES", "FUTURE", "C", "COMMODITY", "COMMODITIES")
        else:
            default_segments = asset_segments_conf
        self._asset_segments: Set[str] = {
            str(segment).upper() for segment in default_segments if segment
        }
        asset_accounts_conf = futures_conf.get("asset_accounts") or futures_conf.get("accounts") or []
        self._asset_accounts: Set[str] = {
            str(account) for account in asset_accounts_conf if account
        }
        self._excluded_segments: Set[str] = {"ALL", "UNIVERSAL", "TOTAL"}

        execution_conf = config.get("execution") or {}
        self._default_time_in_force = execution_conf.get("time_in_force", "DAY")
        self._outside_rth = execution_conf.get("outside_rth")
        self._order_status_poll_interval = float(execution_conf.get("order_status_poll_interval", 1.0))
        self._order_status_retries = int(execution_conf.get("order_status_retries", 3))

    # ------------------------------------------------------------------
    # MarketAdapter interface
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        text = self._canonical_symbol(symbol)
        if not text:
            return text
        cleaned = text
        mapped = self._contract_mapping_active_to_main.get(cleaned)
        if mapped:
            return mapped
        alias = self._alias_lookup.get(cleaned)
        if alias:
            return alias
        if cleaned in self._symbol_meta:
            return cleaned
        prefix = self._extract_contract_prefix(cleaned)
        template_alias = self._prefix_to_template.get(prefix or "")
        if template_alias and template_alias in self._symbol_meta:
            self._alias_lookup[cleaned] = cleaned
            return cleaned
        return cleaned

    async def setup(self) -> None:
        if self._tiger is not None:
            return
        self._tiger = TigerAdapter(self._tiger_conf)
        self.logger.info("Tiger adapter initialized for futures trading")

    async def close(self) -> None:
        self._tiger = None

    async def fetch_account_state(self) -> Dict[str, Any]:
        if self._tiger is None:
            await self.setup()
        assert self._tiger is not None

        # Gather account metrics
        prime_assets = None
        try:
            prime_assets = await self._tiger.get_prime_assets(consolidated=True)
        except Exception:  # pragma: no cover - runtime guard
            self.logger.exception("Failed to fetch Tiger prime assets")

        assets_raw: Iterable[Any] = []
        try:
            assets_raw = await self._tiger.get_assets(segment=True, market_value=True) or []
        except Exception:  # pragma: no cover - runtime guard
            self.logger.exception("Failed to fetch Tiger asset snapshot")

        prime_flat = self._flatten_assets(prime_assets)
        asset_flat = self._flatten_assets(assets_raw)
        prime_list = self._filter_assets(prime_flat)
        asset_list = self._filter_assets(asset_flat)

        if not prime_list:
            prime_list = prime_flat
        if not asset_list:
            asset_list = asset_flat

        if not prime_list and not asset_list:
            self.logger.warning(
                "No Tiger asset snapshot entries matched segments=%s accounts=%s",
                sorted(self._asset_segments) if self._asset_segments else ["*"],
                sorted(self._asset_accounts) if self._asset_accounts else ["*"],
            )

        equity = self._accumulate_numeric(
            prime_list,
            ("net_liquidation",),
            ("summary", "net_liquidation"),
        )
        if equity <= 0:
            equity = self._accumulate_numeric(
                asset_list,
                ("net_liquidation",),
                ("summary", "net_liquidation"),
            )

        available = self._accumulate_numeric(
            prime_list,
            ("available_funds",),
            ("available_cash",),
            ("summary", "available_funds"),
        )
        if available <= 0:
            available = self._accumulate_numeric(
                asset_list,
                ("available_funds",),
                ("summary", "available_funds"),
                ("cash",),
            )

        margin_used = self._accumulate_numeric(
            prime_list,
            ("initial_margin",),
            ("summary", "initial_margin"),
        )
        if margin_used <= 0:
            margin_used = self._accumulate_numeric(
                asset_list,
                ("initial_margin",),
                ("summary", "initial_margin"),
            )
        if margin_used <= 0 and equity > 0 and available >= 0:
            margin_used = max(equity - available, 0.0)

        positions_raw: Iterable[Any] = []
        try:
            positions_raw = await self._tiger.get_positions(sec_type=SecurityType.FUT) or []
        except Exception:  # pragma: no cover - runtime guard
            self.logger.exception("Failed to fetch Tiger futures positions")

        positions_list: List[Dict[str, Any]] = []
        positions_map: Dict[str, Dict[str, Any]] = {}
        total_notional = 0.0

        for position in positions_raw:
            converted = self._convert_position(position)
            if not converted:
                continue
            positions_list.append(converted)
            positions_map[converted["symbol"]] = converted
            mark = ensure_float(converted.get("mark")) or ensure_float(converted.get("entry"))
            contracts = ensure_float(converted.get("contracts"))
            multiplier = ensure_float(converted.get("multiplier"), 1.0)
            total_notional += abs(contracts * multiplier * mark)

        account_state = {
            "net_value": round_float(equity, 2, equity),
            "margin_free": round_float(available, 2, available),
            "margin_used": round_float(margin_used, 2, margin_used),
            "total_pos_notional_value": round_float(total_notional, 2, total_notional),
            "positions": positions_list,
        }

        preview = ", ".join(
            f"{pos['symbol']}:{pos['side']} {pos['contracts']} {pos['upnl']}"
            for pos in positions_list
        ) or "none"
        self.logger.info(
            "Tiger account: equity=%.2f available=%.2f margin=%.2f pos_notional=%.2f positions=%s",
            equity,
            available,
            margin_used,
            total_notional,
            preview,
        )

        return account_state

    @staticmethod
    def _ensure_iterable(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, IterableABC) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _flatten_assets(self, values: Any) -> List[Any]:
        flattened: List[Any] = []
        for item in self._ensure_iterable(values):
            if item is None:
                continue
            sub_accounts = self._extract_path(item, ("sub_accounts",))
            if sub_accounts:
                flattened.extend(self._ensure_iterable(sub_accounts))
            else:
                flattened.append(item)
            segments = self._extract_path(item, ("segments",))
            if segments:
                if isinstance(segments, dict):
                    flattened.extend(self._ensure_iterable(segments.values()))
                else:
                    flattened.extend(self._ensure_iterable(segments))
        return flattened

    def _filter_assets(self, items: Iterable[Any]) -> List[Any]:
        items_list = list(items)
        if not self._asset_segments and not self._asset_accounts:
            return items_list

        filtered: List[Any] = []
        for item in items_list:
            if item is None:
                continue
            segment = self._detect_segment(item)
            account_id = self._detect_account(item)

            segment_match = True
            if self._asset_segments:
                if segment:
                    seg_norm = segment.upper()
                    if seg_norm in self._excluded_segments or seg_norm not in self._asset_segments:
                        segment_match = False
                elif self._asset_accounts and account_id:
                    segment_match = True
                else:
                    segment_match = False

            account_match = True
            if self._asset_accounts:
                if account_id:
                    account_match = account_id in self._asset_accounts
                else:
                    account_match = False

            if segment_match and account_match:
                filtered.append(item)
            elif segment_match and not self._asset_accounts:
                filtered.append(item)
            elif account_match and not self._asset_segments:
                filtered.append(item)
        return filtered

    @staticmethod
    def _extract_path(obj: Any, path: Sequence[str]) -> Any:
        current = obj
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
            if callable(current):  # handle tigeropen descriptors returning callables
                try:
                    current = current()
                except Exception:
                    pass
        return current

    def _detect_segment(self, item: Any) -> Optional[str]:
        for path in (
            ("segment",),
            ("summary", "segment"),
            ("account_type",),
            ("summary", "account_type"),
            ("category",),
            ("summary", "category"),
            ("title",),
            ("summary", "title"),
        ):
            value = self._extract_path(item, path)
            if value is None:
                continue
            if hasattr(value, "value"):
                value = value.value
            elif hasattr(value, "name"):
                value = value.name
            if isinstance(value, str):
                return value
            return str(value)
        return None

    def _detect_account(self, item: Any) -> Optional[str]:
        for path in (
            ("account",),
            ("account_id",),
            ("account_no",),
            ("summary", "account"),
        ):
            value = self._extract_path(item, path)
            if value is None:
                continue
            return str(value)
        return None

    def _describe_assets(self, values: Any) -> List[Dict[str, Any]]:
        described: List[Dict[str, Any]] = []
        for item in self._ensure_iterable(values):
            if item is None:
                continue
            entry: Dict[str, Any] = {}
            for label, path in {
                "account": ("account",),
                "account_id": ("account_id",),
                "segment": ("segment",),
                "account_type": ("account_type",),
                "net_liquidation": ("net_liquidation",),
                "available_funds": ("available_funds",),
                "initial_margin": ("initial_margin",),
            }.items():
                value = self._extract_path(item, path)
                if value is None:
                    continue
                if hasattr(value, "value"):
                    value = value.value
                elif hasattr(value, "name"):
                    value = value.name
                entry[label] = value
            summary = self._extract_path(item, ("summary",))
            if summary:
                summary_dict: Dict[str, Any] = {}
                for key in ("segment", "account", "net_liquidation", "available_funds", "initial_margin"):
                    val = getattr(summary, key, None)
                    if val is None:
                        continue
                    if hasattr(val, "value"):
                        val = val.value
                    elif hasattr(val, "name"):
                        val = val.name
                    summary_dict[key] = val
                if summary_dict:
                    entry["summary"] = summary_dict
            sub_accounts = self._extract_path(item, ("sub_accounts",))
            if sub_accounts:
                entry["sub_accounts"] = len(self._ensure_iterable(sub_accounts))
            described.append(entry)
        return described

    def _accumulate_numeric(self, values: Iterable[Any], *paths: Sequence[str]) -> float:
        total = 0.0
        found = False
        for item in values:
            for path in paths:
                value = self._extract_path(item, path)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(number):
                    continue
                total += number
                found = True
                break
        return total if found else 0.0

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
        if self._tiger is None:
            await self.setup()
        assert self._tiger is not None

        alias = self.normalize_symbol(symbol)
        meta = await self._ensure_metadata_for(alias)
        if not meta:
            self.logger.warning("Symbol %s missing metadata; skipping", alias)
            return None

        data_symbol = meta.get("data_symbol") or alias
        contract_code = meta.get("contract_code") or meta.get("trade_symbol")

        last_close = 0.0

        timeframe_data: Dict[str, Dict[str, Any]] = {}
        last_closed_ts_ms: Optional[int] = None
        primary_seconds = self._timeframe_seconds.get(self._primary_timeframe) or timeframe_to_seconds(self._primary_timeframe) or 3600
        primary_raw_bars: List[List[float]] = []

        for tf in self._timeframes:
            fetch_limit = max(self._fetch_bars.get(tf, 0), 1)
            raw_bars = await self._fetch_future_bars(
                data_symbol,
                contract_code=contract_code,
                period=tf,
                limit=fetch_limit,
            )
            if not raw_bars:
                self.logger.warning("No bars fetched for %s timeframe %s", data_symbol, tf)
                if tf == self._primary_timeframe:
                    return None
                continue

            seconds = self._timeframe_seconds.get(tf) or timeframe_to_seconds(tf) or 0
            closed_bars, closed_ts = trim_to_closed_bars(raw_bars, seconds)
            payload_bars: List[List[float]] = [list(bar) for bar in closed_bars]

            if tf == self._primary_timeframe:
                primary_raw_bars = [list(bar) for bar in raw_bars]
                if not payload_bars or closed_ts is None:
                    return None
                last_closed_ts_ms = closed_ts
                last_close = float(payload_bars[-1][4])
            else:
                last_raw = list(raw_bars[-1])
                if not payload_bars or last_raw[0] != payload_bars[-1][0]:
                    payload_bars.append(last_raw)

            if fetch_limit:
                payload_bars = payload_bars[-max(fetch_limit, 1):]
            if not payload_bars:
                continue

            close_series = [float(item[4]) for item in payload_bars]

            timeframe_data[tf] = {
                "payload_bars": payload_bars,
                "seconds": seconds,
                "close_series": close_series,
            }

        if not timeframe_data or last_closed_ts_ms is None:
            return None

        positions_map = self._positions_to_map(account_state.get("positions"))
        exchange_position = positions_map.get(alias)
        live_price = ensure_float(
            (exchange_position or {}).get("mark")
            or (exchange_position or {}).get("mark_price")
            or (exchange_position or {}).get("price"),
            None,
        )
        if live_price is None and primary_raw_bars:
            candidate = ensure_float(primary_raw_bars[-1][4], None)
            if candidate:
                live_price = candidate
        if live_price is None or live_price <= 0:
            live_price = last_close

        ema_snapshot, rsi_snapshot, macd_snapshot, atr_snapshot = await asyncio.to_thread(
            self._build_indicator_snapshots,
            timeframe_data,
            ema_config=self._ema_config,
            rsi_period=self._rsi_period,
            atr_period=self._atr_period,
            round_price=round_price,
            prepare_macd=self._prepare_macd_details,
        )

        payload_position: Optional[Dict[str, Any]] = None
        if position:
            payload_position = position.to_dict()
        elif exchange_position:
            payload_position = dict(exchange_position)

        bounds = self._build_bounds(meta, live_price)

        timeframes_meta: Dict[str, int] = {
            tf: data_map["seconds"] for tf, data_map in timeframe_data.items()
        }

        payload = {
            "sym": alias,
            "underlying": data_symbol,
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
                self._format_bar(bar) for bar in data_map["payload_bars"][-history_len:]
            ]

        chart_paths: List[str] = []
        if self._chart_enabled and use_chart:
            chart_paths = await asyncio.to_thread(
                self._build_chart_images,
                alias,
                {tf: data_map["payload_bars"] for tf, data_map in timeframe_data.items()},
            )
            if chart_paths:
                payload["chart_images"] = chart_paths

        meta_payload = {
            "data_symbol": data_symbol,
            "trade_symbol": meta.get("trade_symbol") or data_symbol,
            "contract_code": contract_code,
            "sec_type": meta.get("sec_type", "FUT"),
            "exchange": meta.get("exchange"),
            "currency": meta.get("currency"),
            "multiplier": meta.get("multiplier", 1.0),
            "lot_size": meta.get("lot_size"),
            "min_qty": meta.get("min_qty"),
            "initial_margin": meta.get("initial_margin"),
            "maintenance_margin": meta.get("maintenance_margin"),
             "max_qty": meta.get("max_qty"),
            "bounds": bounds,
            "chart_images": chart_paths,
            "suggested_interval_seconds": timeframes_meta.get(self._primary_timeframe, primary_seconds),
            "timeframes": timeframes_meta,
        }

        return SymbolData(
            symbol=alias,
            payload=payload,
            last_ts=last_closed_ts_ms,
            last_price=live_price,
            meta=meta_payload,
            position=position,
        )

    async def execute_decision(
        self,
        symbol: str,
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._tiger is None:
            await self.setup()
        assert self._tiger is not None

        action = str(decision.get("action") or decision.get("act") or "").upper()
        if not action:
            return None

        non_order_actions = {
            "WAIT",
            "HOLD",
            "NONE",
            "ADJUST_SL",
            "ADJUST_TP",
            "TRAIL_ON",
            "TRAIL_OFF",
            "RISK_OFF",
        }
        if action in non_order_actions:
            return None

        qty_raw = decision.get("amount")
        if qty_raw is None:
            self.logger.warning("Decision missing quantity: %s", decision)
            return None
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError):
            self.logger.warning("Invalid quantity in decision: %s", decision)
            return None
        if qty <= 0:
            self.logger.warning("Non-positive quantity in decision: %s", decision)
            return None

        alias = self.normalize_symbol(symbol)
        meta = await self._ensure_metadata_for(alias)
        if not meta:
            self.logger.warning("Decision for unknown symbol %s", alias)
            return None

        positions_map = self._positions_to_map(account_state.get("positions"))
        current_pos = positions_map.get(alias)
        current_amount_abs = 0.0
        net_current = 0.0
        if current_pos:
            raw_amount = ensure_float(
                current_pos.get("amount")
                if current_pos.get("amount") is not None
                else current_pos.get("contracts")
            )
            if raw_amount is not None:
                current_amount_abs = abs(raw_amount)
                pos_side = str(current_pos.get("side") or "long").lower()
                net_current = current_amount_abs if pos_side != "short" else -current_amount_abs

        trade_symbol = meta.get("trade_symbol") or alias
        sec_type = meta.get("sec_type", "FUT")
        exchange = meta.get("exchange")
        currency = meta.get("currency")
        contract_code = meta.get("contract_code") or trade_symbol

        quantity = self._apply_lot_step(qty, meta)
        if quantity <= 0:
            self.logger.warning("Quantity rounded to zero for %s", alias)
            return None

        side = self._resolve_side(action, current_pos, constraints=self.config.get("constraints", {}))
        if side is None:
            self.logger.info("No executable side for action=%s symbol=%s", action, alias)
            return None

        if action == "CLOSE":
            if not current_pos:
                return None
            quantity = ensure_float(current_pos.get("contracts"))
            quantity = self._apply_lot_step(quantity, meta)
        elif action in {"REDUCE", "PARTIAL_TP"}:
            if not current_pos:
                return None
            max_qty = self._apply_lot_step(ensure_float(current_pos.get("contracts")), meta)
            quantity = min(quantity, max_qty)
            quantity = self._apply_lot_step(quantity, meta)
        quantity = self._apply_lot_step(quantity, meta)

        quantity = self._adjust_quantity_for_limits(
            symbol=alias,
            quantity=quantity,
            side=side,
            action=action,
            net_current=net_current,
            meta=meta,
            data=data,
            decision=decision,
            account_state=account_state,
        )
        if quantity <= 0:
            self.logger.info(
                "Skip %s: quantity constrained to zero after risk checks (action=%s)",
                alias,
                action,
            )
            return None
        decision["amount"] = quantity

        tp_threshold = self._max_amount_per_symbol if self._max_amount_per_symbol > 0 else 1.0
        if action in {"BUY", "ADD", "SELL"} and quantity <= max(tp_threshold, 1.0):
            decision["tp_policy"] = "trailing"
            trailing_conf = decision.get("trailing") or decision.get("trail")
            if not isinstance(trailing_conf, dict) or not trailing_conf.get("kind"):
                decision["trailing"] = {
                    "kind": "pct",
                    "dist": 0.01,
                }

        if quantity <= 0:
            return None

        order = await self._submit_market_order(
            trade_symbol,
            side=side,
            quantity=quantity,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            contract_code=contract_code,
        )
        if order is None:
            return None

        order_dict = order if isinstance(order, dict) else self._order_to_dict(order)
        oid = order_dict.get('id')
        filled = order_dict.get('filled', 0)
        self.logger.info(
            f"Order result: {alias}, id={oid}, action={action}, side={side}, requested={quantity}, filled={filled}"
        )
        return {
            "symbol": alias,
            "id": oid,
            "action": action,
            "side": side,
            "requested_qty": quantity,
            "filled": filled,
            "order": order_dict,
            "order_type": "MARKET",
            "trade_symbol": trade_symbol,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_symbol_metadata(self, futures_conf: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        symbol_overrides = futures_conf.get("symbol_overrides") or {}
        meta: Dict[str, Dict[str, Any]] = {}
        alias_lookup: Dict[str, str] = {}
        for alias, payload in symbol_overrides.items():
            alias_u = str(alias).upper().strip()
            if not alias_u:
                continue
            info = dict(payload or {})
            data_symbol = str(info.get("data_symbol") or info.get("contract_code") or alias_u).upper()
            trade_symbol = str(info.get("trade_symbol") or info.get("contract_code") or alias_u).upper()
            info.setdefault("sec_type", "FUT")
            info.setdefault("currency", "USD")
            info.setdefault("lot_size", info.get("lot_step", 1.0) or 1.0)
            info.setdefault("min_qty", info.get("lot_size", 1.0))
            info.setdefault("multiplier", info.get("multiplier", 1.0))
            info["data_symbol"] = data_symbol
            info.setdefault("trade_symbol", trade_symbol)
            max_qty_raw = info.get("max_qty")
            if max_qty_raw is not None:
                try:
                    info["max_qty"] = float(max_qty_raw)
                except (TypeError, ValueError):
                    info.pop("max_qty", None)
            meta[alias_u] = info
            alias_lookup[alias_u] = alias_u
            alias_lookup[data_symbol] = alias_u
            alias_lookup[trade_symbol] = alias_u
            for alt in info.get("aliases", []) or []:
                alias_lookup[str(alt).upper()] = alias_u
        return meta, alias_lookup

    def _load_contract_mapping(self, futures_conf: Dict[str, Any]) -> None:
        mapping_conf = futures_conf.get("contract_mapping") or {}
        for main, active in mapping_conf.items():
            main_clean = self._canonical_symbol(main)
            active_clean = self._canonical_symbol(active)
            if not main_clean or not active_clean:
                continue
            self._contract_mapping_main_to_active[main_clean] = active_clean
            self._contract_mapping_active_to_main[active_clean] = main_clean
            self._alias_lookup[active_clean] = main_clean
            info = self._symbol_meta.get(main_clean)
            if info:
                info["contract_code"] = active_clean
                info["trade_symbol"] = active_clean
                info["data_symbol"] = active_clean
                aliases = list(info.get("aliases") or [])
                if active_clean not in aliases:
                    aliases.append(active_clean)
                info["aliases"] = aliases

    def _build_prefix_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for alias, info in self._symbol_meta.items():
            for prefix in self._derive_prefixes(alias, info):
                if prefix and prefix not in mapping:
                    mapping[prefix] = alias
        return mapping

    def _derive_prefixes(self, alias: str, info: Dict[str, Any]) -> Set[str]:
        prefixes: Set[str] = set()
        explicit = info.get("contract_prefix")
        if explicit:
            prefixes.add(str(explicit).upper())
        for key in (
            alias,
            str(info.get("data_symbol") or ""),
            str(info.get("trade_symbol") or ""),
        ):
            if not key:
                continue
            key_up = key.upper()
            if key_up.endswith("MAIN") and len(key_up) > 4:
                prefixes.add(key_up[:-4])
            inferred = self._extract_contract_prefix(key_up)
            if inferred:
                prefixes.add(inferred)
        for alt in info.get("aliases", []) or []:
            inferred = self._extract_contract_prefix(str(alt).upper())
            if inferred:
                prefixes.add(inferred)
        return {p for p in prefixes if p}

    async def _ensure_metadata_for(self, symbol: str) -> Optional[Dict[str, Any]]:
        alias = self._canonical_symbol(symbol)
        existing = self._symbol_meta.get(alias)
        if existing and not self._metadata_requires_enrichment(existing):
            return existing

        base_template = None
        if existing is None:
            base_template = self._derive_metadata_from_template(alias)

        base_meta = dict(existing) if existing else (dict(base_template) if base_template else None)
        fetched = await self._maybe_fetch_contract_metadata(alias, base_meta)
        if fetched:
            combined = self._merge_metadata(base_meta, fetched)
        else:
            if existing is not None:
                return existing
            combined = base_meta

        if combined is None:
            if existing:
                return existing
            if base_template:
                return self._register_metadata(alias, base_template)
            return None

        return self._register_metadata(alias, combined)

    def _metadata_requires_enrichment(self, meta: Dict[str, Any]) -> bool:
        if not meta:
            return True
        if meta.get("_source") == "tiger_contract":
            return False
        return True

    def _derive_metadata_from_template(self, symbol: str) -> Optional[Dict[str, Any]]:
        prefix = self._extract_contract_prefix(symbol)
        template_key = self._prefix_to_template.get(prefix or "")
        template = self._symbol_meta.get(template_key) if template_key else None
        if not template:
            return None
        derived = dict(template)
        aliases = list(template.get("aliases") or [])
        if symbol not in aliases:
            aliases.append(symbol)
        derived["aliases"] = aliases
        derived["contract_code"] = symbol
        derived["trade_symbol"] = symbol
        derived["data_symbol"] = symbol
        return derived

    def _merge_metadata(
        self,
        base: Optional[Dict[str, Any]],
        updates: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if base is None and not updates:
            return None
        merged: Dict[str, Any] = {}
        if base:
            merged.update(base)
            if "aliases" in base:
                merged["aliases"] = list(base.get("aliases") or [])
        if not updates:
            return merged
        aliases = list(updates.get("aliases") or [])
        for key, value in updates.items():
            if key == "aliases":
                continue
            if key == "_source":
                merged[key] = value
                continue
            if self._is_missing_metadata_value(merged.get(key)):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        if aliases:
            merged.setdefault("aliases", [])
            merged["aliases"].extend(aliases)
        if "_source" in updates and updates["_source"]:
            merged["_source"] = updates["_source"]
        return merged

    def _register_metadata(self, alias: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        alias_u = self._canonical_symbol(alias)
        info = dict(payload)

        for key in ("data_symbol", "trade_symbol", "contract_code"):
            value = info.get(key)
            if isinstance(value, str):
                info[key] = self._canonical_symbol(value) or alias_u
        sec_type_val = info.get("sec_type")
        if hasattr(sec_type_val, "value"):
            info["sec_type"] = sec_type_val.value
        elif isinstance(sec_type_val, str):
            info["sec_type"] = sec_type_val.upper()

        lot_size_val = ensure_float(info.get("lot_size"), None)
        lot_step_val = ensure_float(info.get("lot_step"), None)
        if lot_size_val is None and lot_step_val is not None:
            info["lot_size"] = lot_step_val
        if lot_step_val is None and lot_size_val is not None:
            info["lot_step"] = lot_size_val
        if self._is_missing_metadata_value(info.get("min_qty")):
            info["min_qty"] = info.get("lot_size") or 1.0

        alias_set: Set[str] = {alias_u}
        for key in ("data_symbol", "trade_symbol", "contract_code"):
            value = info.get(key)
            if isinstance(value, str):
                cleaned = self._canonical_symbol(value)
                if cleaned:
                    alias_set.add(cleaned)
        for alt in info.get("aliases", []) or []:
            if isinstance(alt, str):
                cleaned = self._canonical_symbol(alt)
                if cleaned:
                    alias_set.add(cleaned)
        info["aliases"] = sorted(alias_set)

        self._symbol_meta[alias_u] = info
        for alt in info["aliases"]:
            self._alias_lookup[alt] = alias_u

        for prefix in self._derive_prefixes(alias_u, info):
            if prefix and prefix not in self._prefix_to_template:
                self._prefix_to_template[prefix] = alias_u
        return info

    async def _maybe_fetch_contract_metadata(
        self,
        symbol: str,
        base_meta: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        alias = self._canonical_symbol(symbol)
        if alias in self._metadata_fetch_failures:
            return None
        if base_meta and not self._metadata_requires_enrichment(base_meta):
            return None
        fetched = await self._fetch_contract_metadata(alias, base_meta)
        if not fetched:
            self._metadata_fetch_failures.add(alias)
            return None
        return fetched

    async def _fetch_contract_metadata(
        self,
        symbol: str,
        base_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self._tiger is None:
            await self.setup()
        assert self._tiger is not None
        candidates: List[str] = [symbol]
        mapped = self._contract_mapping_main_to_active.get(symbol)
        if mapped and mapped not in candidates:
            candidates.append(mapped)
        if base_meta:
            for key in ("contract_code", "trade_symbol", "data_symbol"):
                candidate = base_meta.get(key)
                if not isinstance(candidate, str):
                    continue
                cleaned = self._canonical_symbol(candidate)
                if cleaned and cleaned not in candidates:
                    candidates.append(cleaned)

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                contract = await self._tiger.get_contract(
                    candidate,
                    sec_type=SecurityType.FUT if SecurityType is not None else None,
                )
            except Exception as exc:  # pragma: no cover - network/runtime errors
                last_error = exc
                continue
            if contract is None:
                continue
            meta = self._contract_to_metadata(contract, candidate)
            meta["_source"] = "tiger_contract"
            return meta

        if last_error is not None:
            self.logger.error("Failed to fetch contract metadata for %s", symbol, exc_info=last_error)
        return None

    def _contract_to_metadata(self, contract: Any, fallback: str) -> Dict[str, Any]:
        identifier = getattr(contract, "identifier", None)
        local_symbol = getattr(contract, "local_symbol", None)
        base_symbol = getattr(contract, "symbol", None)

        data_symbol = self._canonical_symbol(base_symbol or identifier or fallback)
        trade_symbol = self._canonical_symbol(local_symbol or identifier or base_symbol or fallback)
        contract_code = self._canonical_symbol(identifier or local_symbol or base_symbol or fallback)

        multiplier = ensure_float(getattr(contract, "multiplier", None), None)
        if multiplier is None or multiplier <= 0:
            multiplier = 1.0
        lot_size = ensure_float(getattr(contract, "lot_size", None), None)
        if lot_size is None or lot_size <= 0:
            lot_size = 1.0

        initial_margin = ensure_float(getattr(contract, "long_initial_margin", None), None)
        maintenance_margin = ensure_float(getattr(contract, "long_maintenance_margin", None), None)

        aliases = {
            data_symbol,
            trade_symbol,
            contract_code,
            self._canonical_symbol(fallback),
        }

        meta = {
            "data_symbol": data_symbol,
            "trade_symbol": trade_symbol,
            "contract_code": contract_code,
            "sec_type": getattr(contract, "sec_type", "FUT") or "FUT",
            "exchange": getattr(contract, "exchange", None),
            "currency": getattr(contract, "currency", None),
            "multiplier": multiplier,
            "lot_size": lot_size,
            "min_qty": ensure_float(getattr(contract, "lot_size", None), None) or lot_size,
            "initial_margin": initial_margin,
            "maintenance_margin": maintenance_margin,
            "min_tick": ensure_float(getattr(contract, "min_tick", None), None),
            "expiry": getattr(contract, "expiry", None),
            "first_notice_date": getattr(contract, "first_notice_date", None),
            "last_trading_date": getattr(contract, "last_trading_date", None),
            "continuous": getattr(contract, "continuous", None),
            "identifier": identifier,
            "name": getattr(contract, "name", None),
            "origin_symbol": getattr(contract, "origin_symbol", None),
            "aliases": list(alias for alias in aliases if alias),
        }
        support_overnight = getattr(contract, "support_overnight_trading", None)
        if support_overnight is not None:
            meta["support_overnight_trading"] = support_overnight
        return meta

    @staticmethod
    def _is_missing_metadata_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (int, float)):
            return value == 0
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _extract_contract_prefix(symbol: Optional[str]) -> Optional[str]:
        if not symbol:
            return None
        symbol = str(symbol).upper()
        prefix_chars: List[str] = []
        for ch in symbol:
            if ch.isalpha():
                prefix_chars.append(ch)
            else:
                break
        if prefix_chars:
            return "".join(prefix_chars)
        return None

    @staticmethod
    def _canonical_symbol(symbol: Optional[str]) -> str:
        if not symbol:
            return ""
        return str(symbol).upper().replace(":", "").replace("/", "")

    async def is_market_open(self, symbol: str) -> bool:
        alias = self.normalize_symbol(symbol)
        identifier = self._resolve_active_identifier(alias)
        if not identifier:
            return True
        now = time.time()
        cached = self._market_status_cache.get(identifier)
        if cached and now - cached[0] < self._market_status_ttl:
            return cached[1]
        is_open = await self._compute_market_open(identifier)
        self._market_status_cache[identifier] = (now, is_open)
        return is_open

    def _resolve_active_identifier(self, alias: str) -> Optional[str]:
        meta = self._symbol_meta.get(alias)
        if not meta:
            return None
        contract = meta.get("contract_code") or meta.get("trade_symbol") or meta.get("data_symbol")
        return self._canonical_symbol(contract)

    async def _compute_market_open(self, identifier: str) -> bool:
        if self._tiger is None:
            await self.setup()
        assert self._tiger is not None
        now_ms = int(time.time() * 1000)
        try:
            trading_times = await self._tiger.get_future_trading_times(identifier)
        except Exception:
            self.logger.exception("Failed to fetch trading times for %s", identifier)
            return True
        if trading_times is None:
            return True
        sessions: Iterable[Any]
        if isinstance(trading_times, pd.DataFrame):
            if trading_times.empty:
                return True
            sessions = trading_times.to_dict(orient="records")
        elif isinstance(trading_times, dict):
            sessions = trading_times.get("items") or trading_times.get("data") or []
        else:
            sessions = trading_times
        for session in self._ensure_iterable(sessions):
            if session is None:
                continue
            if isinstance(session, dict):
                start = ensure_float(session.get("start"), 0.0)
                end = ensure_float(session.get("end"), 0.0)
            else:
                start = ensure_float(_safe_attr(session, "start"), 0.0)
                end = ensure_float(_safe_attr(session, "end"), 0.0)
            if start and end and start <= now_ms <= end:
                return True
        return False

    @staticmethod
    def _canonical_symbol(symbol: Optional[str]) -> str:
        if not symbol:
            return ""
        return str(symbol).upper().replace(":", "").replace("/", "")

    async def _fetch_future_bars(
        self,
        data_symbol: str,
        *,
        contract_code: Optional[str],
        period: Any = None,
        limit: Optional[int] = None,
    ) -> List[List[float]]:
        assert self._tiger is not None
        bar_period_raw = period or (getattr(BarPeriod, "HOUR", "HOUR") if BarPeriod is not None else "HOUR")
        bar_period = self._normalize_future_period(bar_period_raw)
        try:
            bars = await self._tiger.get_future_bars(
                identifier=str(contract_code or data_symbol),
                period=bar_period,
                begin_time=-1,
                end_time=-1,
                limit=limit,
            )
        except Exception:  # pragma: no cover - runtime guard
            self.logger.exception("Failed to fetch future bars for %s", data_symbol)
            return []
        return self._convert_bars(bars)

    @staticmethod
    def _normalize_future_period(value: Any) -> str:
        if value is None:
            return "60min"
        if hasattr(value, "value"):
            value = value.value
        text = str(value).strip().lower()
        mapping = {
            "hour": "60min",
            "1h": "60min",
            "1hour": "60min",
            "60": "60min",
            "60m": "60min",
            "60min": "60min",
            "day": "day",
            "d": "day",
            "daily": "day",
            "week": "week",
            "monthly": "month",
            "month": "month",
            "4h": "4hour",
            "4hour": "4hour",
            "240": "4hour",
        }
        return mapping.get(text, text or "60min")

    def _convert_bars(self, bars: Any) -> List[List[float]]:
        if bars is None:
            return []
        if isinstance(bars, pd.DataFrame):
            if bars.empty:
                return []
            items_iter: Iterable[Any] = bars.to_dict(orient="records")
        elif isinstance(bars, dict):
            items_iter = bars.get("items") or bars.get("data") or []
        else:
            items_iter = bars
        if not items_iter:
            return []
        converted: List[List[float]] = []
        for item in items_iter:
            if item is None:
                continue
            if isinstance(item, dict):
                open_time = item.get("time") or item.get("timestamp") or item.get("begin_time")
                open_price = item.get("open") or item.get("open_price")
                high = item.get("high") or item.get("high_price")
                low = item.get("low") or item.get("low_price")
                close = item.get("close") or item.get("close_price")
                volume = item.get("volume") or item.get("amount")
            else:
                open_time = _safe_attr(item, "time") or _safe_attr(item, "timestamp") or _safe_attr(item, "begin_time")
                open_price = _safe_attr(item, "open") or _safe_attr(item, "open_price")
                high = _safe_attr(item, "high") or _safe_attr(item, "high_price")
                low = _safe_attr(item, "low") or _safe_attr(item, "low_price")
                close = _safe_attr(item, "close") or _safe_attr(item, "close_price")
                volume = _safe_attr(item, "volume") or _safe_attr(item, "amount")
            if open_time is None or open_price is None or close is None:
                continue
            open_time_ms = float(open_time)
            if open_time_ms < 10_000_000_000:
                open_time_ms *= 1000.0
            record = [
                int(open_time_ms),
                ensure_float(open_price),
                ensure_float(high, ensure_float(open_price)),
                ensure_float(low, ensure_float(open_price)),
                ensure_float(close),
                ensure_float(volume),
            ]
            converted.append(record)
        converted.sort(key=lambda row: row[0])
        return converted

    def _prepare_macd_details(self, close_series: Sequence[float]) -> Dict[str, Any]:
        if not close_series:
            return {"macd": None, "signal": None, "hist": None}
        macd_result = macd_series(close_series)
        macd_line = getattr(macd_result, "macd", None) or []
        signal_line = getattr(macd_result, "signal", None) or []
        hist_line = getattr(macd_result, "histogram", None) or getattr(macd_result, "hist", None) or []
        if not macd_line and not signal_line and not hist_line:
            return {"macd": None, "signal": None, "hist": None}
        return {
            "macd": self._round_numeric_series(macd_line, count=5, decimals=4) if macd_line else None,
            "signal": self._round_numeric_series(signal_line, count=5, decimals=4) if signal_line else None,
            "hist": self._round_numeric_series(hist_line, count=5, decimals=4) if hist_line else None,
        }

    def _build_bounds(self, meta: Dict[str, Any], last_price: float) -> Dict[str, Any]:
        lot_step = ensure_float(meta.get("lot_size") or meta.get("lot_step") or 1.0, 1.0)
        min_qty = ensure_float(meta.get("min_qty"), lot_step)
        max_qty = meta.get("max_qty")
        if max_qty is not None:
            max_qty = ensure_float(max_qty)
            if max_qty is not None and max_qty <= 0:
                max_qty = None
        return {
            "qty_abs": [min_qty, max_qty],
            "lot_step": lot_step,
            "min_notional": None,
            "max_notional": None,
        }

    def _format_bar(self, bar: Sequence[Any]) -> List[Any]:
        if len(bar) < 6:
            return list(bar)
        ts_sec = int(float(bar[0]) / 1000)
        return [
            ensure_float(bar[1]),
            ensure_float(bar[2]),
            ensure_float(bar[3]),
            ensure_float(bar[4]),
            ensure_float(bar[5]),
            ts_sec,
        ]

    def _positions_to_map(self, positions: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if not positions:
            return result
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            symbol = pos.get("symbol") or pos.get("sym")
            if not symbol:
                continue
            normalized = self.normalize_symbol(str(symbol))
            result[normalized] = dict(pos)
        return result

    def _convert_position(self, position: Any) -> Optional[Dict[str, Any]]:
        contract = getattr(position, "contract", None)
        raw_symbol = (
            getattr(contract, "symbol", None)
            or getattr(contract, "local_symbol", None)
            or getattr(position, "symbol", None)
        )
        if not raw_symbol:
            return None
        alias = self.normalize_symbol(raw_symbol)
        quantity = ensure_float(
            getattr(position, "quantity", None),
            ensure_float(getattr(position, "position", None), 0.0),
        )
        side = "long" if quantity >= 0 else "short"
        contracts = abs(quantity)
        avg_cost = ensure_float(
            getattr(position, "average_cost", None),
            ensure_float(getattr(position, "avg_cost", None), 0.0),
        )
        mark = ensure_float(
            getattr(position, "market_price", None),
            ensure_float(getattr(position, "last_price", None), avg_cost),
        )
        upnl = ensure_float(
            getattr(position, "unrealized_pnl", None),
            ensure_float(getattr(position, "unrealized", None), 0.0),
        )
        multiplier = ensure_float(
            getattr(contract, "multiplier", None),
            ensure_float(getattr(position, "multiplier", None), 1.0),
        )
        return {
            "symbol": alias,
            "side": side,
            "contracts": contracts,
            "amount": contracts,
            "entry": round_price(avg_cost),
            "entry_price": round_price(avg_cost),
            "mark": round_price(mark),
            "upnl": round_float(upnl, 2, upnl),
            "multiplier": multiplier,
        }

    def _apply_lot_step(self, quantity: float, meta: Dict[str, Any]) -> float:
        step = ensure_float(meta.get("lot_size") or meta.get("lot_step") or 1.0, 1.0)
        if step <= 0:
            return max(quantity, 0.0)
        return math.floor(quantity / step + 1e-9) * step

    def _adjust_quantity_for_limits(
        self,
        *,
        symbol: str,
        quantity: float,
        side: str,
        action: str,
        net_current: float,
        meta: Dict[str, Any],
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> float:
        quantity = max(quantity, 0.0)
        if quantity <= 0:
            return 0.0

        original_qty = quantity
        quantity = self._apply_lot_step(quantity, meta)
        if quantity <= 0:
            return 0.0

        adjusted_by_margin = self._clamp_by_margin(
            quantity,
            net_current,
            side,
            meta,
            data,
            decision,
            account_state,
        )
        if adjusted_by_margin < quantity:
            self.logger.info(
                "Adjusted quantity by margin constraints: symbol=%s requested=%.4f adjusted=%.4f",
                symbol,
                quantity,
                adjusted_by_margin,
            )
        quantity = adjusted_by_margin
        if quantity <= 0:
            return 0.0

        cap_limit = self._resolve_position_cap(meta)
        if cap_limit is not None and cap_limit > 0:
            adjusted_by_cap = self._clamp_by_position_caps(
                quantity,
                net_current,
                side,
                cap_limit,
                meta,
            )
            if adjusted_by_cap < quantity:
                self.logger.info(
                    "Adjusted quantity by cap constraints: symbol=%s requested=%.4f adjusted=%.4f cap=%.4f",
                    symbol,
                    quantity,
                    adjusted_by_cap,
                    cap_limit,
                )
            quantity = adjusted_by_cap

        if quantity <= 0:
            return 0.0
        if not math.isclose(quantity, original_qty, rel_tol=0.0, abs_tol=1e-9):
            quantity = self._apply_lot_step(quantity, meta)
        return max(quantity, 0.0)

    def _clamp_by_margin(
        self,
        quantity: float,
        net_current: float,
        side: str,
        meta: Dict[str, Any],
        data: SymbolData,
        decision: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> float:
        close_part = self._estimate_close_part(quantity, net_current, side)
        increase_part = max(quantity - close_part, 0.0)
        if increase_part <= 0:
            return quantity

        margin_per_contract = self._estimate_margin_per_contract(meta, data, decision)
        if margin_per_contract <= 0:
            return quantity

        margin_free = ensure_float(account_state.get("margin_free"), None)
        if margin_free is None:
            margin_free = ensure_float(account_state.get("available_funds"), None)
        if margin_free is None:
            return quantity
        net_value = ensure_float(account_state.get("net_value"), None)
        if net_value is None:
            net_value = ensure_float(account_state.get("equity"), 0.0)

        margin_floor = max(
            self._margin_free_floor_abs,
            self._margin_free_floor_pct * net_value if net_value and net_value > 0 else 0.0,
        )
        available_margin = margin_free - margin_floor
        if available_margin <= 0:
            return self._apply_lot_step(close_part, meta)

        max_additional = math.floor(available_margin / margin_per_contract + 1e-9)
        max_additional = self._apply_lot_step(max_additional, meta)
        if max_additional <= 0:
            return self._apply_lot_step(close_part, meta)

        if increase_part <= max_additional + 1e-9:
            return quantity

        new_quantity = close_part + max_additional
        new_quantity = min(new_quantity, quantity)
        return self._apply_lot_step(max(new_quantity, 0.0), meta)

    def _resolve_position_cap(self, meta: Dict[str, Any]) -> Optional[float]:
        candidates: List[float] = []
        if self._max_amount_per_symbol and self._max_amount_per_symbol > 0:
            candidates.append(self._max_amount_per_symbol)
        meta_cap = ensure_float(meta.get("max_qty"), None)
        if meta_cap and meta_cap > 0:
            candidates.append(meta_cap)
        if not candidates:
            return None
        return min(candidates)

    def _clamp_by_position_caps(
        self,
        quantity: float,
        net_current: float,
        side: str,
        cap_limit: float,
        meta: Dict[str, Any],
    ) -> float:
        desired_delta = quantity if side == "BUY" else -quantity
        projected = net_current + desired_delta
        upper_limit = cap_limit
        lower_limit = -cap_limit if not self._long_only else 0.0

        if lower_limit - 1e-9 <= projected <= upper_limit + 1e-9:
            return quantity

        close_part = self._estimate_close_part(quantity, net_current, side)

        allowed_projected = projected
        if projected > upper_limit:
            allowed_projected = upper_limit
        elif projected < lower_limit:
            allowed_projected = lower_limit

        allowed_delta = allowed_projected - net_current
        if desired_delta > 0 and allowed_delta <= 0:
            return self._apply_lot_step(close_part, meta)
        if desired_delta < 0 and allowed_delta >= 0:
            return self._apply_lot_step(close_part, meta)

        new_quantity = abs(allowed_delta)
        if close_part > 0 and new_quantity < close_part:
            new_quantity = close_part
        new_quantity = min(new_quantity, quantity)
        return self._apply_lot_step(max(new_quantity, 0.0), meta)

    @staticmethod
    def _estimate_close_part(quantity: float, net_current: float, side: str) -> float:
        if net_current > 0 and side == "SELL":
            return min(quantity, net_current)
        if net_current < 0 and side == "BUY":
            return min(quantity, abs(net_current))
        return 0.0

    def _estimate_margin_per_contract(
        self,
        meta: Dict[str, Any],
        data: SymbolData,
        decision: Dict[str, Any],
    ) -> float:
        margin = ensure_float(meta.get("initial_margin"), None)
        if margin is None or margin <= 0:
            margin = ensure_float(meta.get("maintenance_margin"), None)
        if margin is not None and margin > 0:
            return margin * (1.0 + self._margin_buffer_pct) if self._margin_buffer_pct > 0 else margin

        price = ensure_float(decision.get("price") or decision.get("entry"), None)
        if price is None or price <= 0:
            price = ensure_float(getattr(data, "last_price", None), None)
        multiplier = ensure_float(meta.get("multiplier"), 1.0)
        if price is None or price <= 0 or multiplier <= 0:
            return 0.0

        ratio = 0.1 # default 10x leverage if cannot get contract metadata

        margin = price * multiplier * ratio
        if self._margin_buffer_pct > 0:
            margin *= 1.0 + self._margin_buffer_pct
        return margin

    def _resolve_side(self, action: str, position: Optional[Dict[str, Any]], *, constraints: Dict[str, Any]) -> Optional[str]:
        action_u = action.upper()
        long_only = bool(constraints.get("long_only"))
        if action_u in {"BUY", "ADD"}:
            return "BUY"
        if action_u == "SELL":
            return None if long_only else "SELL"
        if action_u in {"REDUCE", "PARTIAL_TP", "CLOSE"}:
            if not position:
                return None
            side = str(position.get("side") or "long").lower()
            if side == "short":
                return "BUY"
            return "SELL"
        return None

    async def _submit_market_order(
        self,
        trade_symbol: str,
        *,
        side: str,
        quantity: float,
        sec_type: Any,
        exchange: Optional[str],
        currency: Optional[str],
        contract_code: Optional[str],
    ) -> Any:
        assert self._tiger is not None
        try:
            order = await self._tiger.place_market_order(
                contract_code or trade_symbol,
                side=side,
                quantity=quantity,
                sec_type=sec_type,
                currency=currency,
                exchange=exchange,
                time_in_force=self._default_time_in_force,
                outside_rth=self._outside_rth,
            )
            if order is None:
                return None
            order_dict = order if isinstance(order, dict) else self._order_to_dict(order)
            return await self._refresh_order_status(order_dict)
        except Exception:  # pragma: no cover - runtime guard
            self.logger.exception(
                "Tiger market order failed: symbol=%s side=%s qty=%s",
                trade_symbol,
                side,
                quantity,
            )
            return None

    def _order_to_dict(self, order: Any) -> Dict[str, Any]:
        if order is None:
            return {}
        attrs = {
            "id": order.id,
            "order_id": order.order_id,
            "parent_id": getattr(order, "parent_id", None),
            "order_time": getattr(order, "order_time", None),
            "trade_time": getattr(order, "trade_time", None),
            "update_time": getattr(order, "update_time", None),
            "action": order.action,
            "order_type": getattr(order, "order_type", None),
            "status": getattr(order, "status", None),
            "quantity": getattr(order, "quantity", None),
            "filled": getattr(order, "filled", None),
            "remaining": getattr(order, "remaining", None),
            "avg_fill_price": getattr(order, "avg_fill_price", None),
            "limit_price": getattr(order, "limit_price", None),
            "aux_price": getattr(order, "aux_price", None),
            "commission": getattr(order, "commission", None),
            "realized_pnl": getattr(order, "realized_pnl", None),
            "time_in_force": getattr(order, "time_in_force", None),
            "outside_rth": getattr(order, "outside_rth", None),
            "source": getattr(order, "source", None),
            "account": getattr(order, "account", None),
            "symbol": getattr(order, "symbol", None),
        }
        compact = {key: self._serialize_value(value) for key, value in attrs.items() if value is not None}

        contract = getattr(order, "contract", None)
        if contract is not None:
            compact["contract"] = self._serialize_value(contract)
            contract_symbol = getattr(contract, "symbol", None) or getattr(contract, "local_symbol", None)
            if contract_symbol and "symbol" not in compact:
                contract_symbol_text = self._serialize_value(contract_symbol)
                if isinstance(contract_symbol_text, str):
                    compact["symbol"] = contract_symbol_text.split("/")[0]
                else:
                    compact["symbol"] = contract_symbol_text
        charges = getattr(order, "charges", None)
        if charges:
            compact["charges"] = self._serialize_value(charges)
        return compact

    def _serialize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "value"):
            try:
                return value.value
            except Exception:
                pass
        if hasattr(value, "name"):
            try:
                return value.name
            except Exception:
                pass
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return str(value)

    async def _refresh_order_status(self, order: Dict[str, Any]) -> Dict[str, Any]:
        order_id_val = order.get("order_id")
        global_id_val = order.get("id")
        order_id_int: Optional[int] = None
        global_id_int: Optional[int] = None
        try:
            if order_id_val not in (None, "", 0):
                order_id_int = int(order_id_val)
        except (TypeError, ValueError):
            order_id_int = None
        try:
            if global_id_val not in (None, "", 0):
                global_id_int = int(global_id_val)
        except (TypeError, ValueError):
            global_id_int = None
        if order_id_int is not None and order_id_int >= 10 ** 12:
            global_id_int = order_id_int
            order_id_int = None
        if order_id_int is None and global_id_int is None:
            return order

        latest = dict(order)
        retries = max(1, self._order_status_retries)
        interval = max(0.1, self._order_status_poll_interval)
        for attempt in range(retries):
            if attempt > 0:
                await asyncio.sleep(interval)
            try:
                refreshed = await self._tiger.get_order(
                    order_id=order_id_int,
                    global_id=global_id_int,
                )
            except Exception:
                identifier = order_id_int or global_id_int
                self.logger.exception("Failed to refresh order status for %s (attempt %s)", identifier, attempt + 1)
                continue
            if not refreshed:
                continue
            latest = refreshed if isinstance(refreshed, dict) else self._order_to_dict(refreshed)
            status = str(latest.get("status") or "").upper()
            filled_qty = ensure_float(
                latest.get("filled_quantity")
                or latest.get("filled")
                or latest.get("executedQty")
                or latest.get("executed_quantity")
            )
            if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED"} or (filled_qty is not None and filled_qty > 0):
                break
        return latest


__all__ = ["TigerFuturesAdapter"]
