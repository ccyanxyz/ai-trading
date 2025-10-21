"""Common helpers for AI strategies."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


def ensure_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_float(
    payload: Mapping[str, Any],
    keys: Iterable[str],
    *,
    default: Optional[float] = None,
    require_positive: bool = False,
) -> Optional[float]:
    """Return the first convertible float in *payload* under the given *keys*."""

    if not hasattr(payload, "get"):
        return default

    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if require_positive and number <= 0:
            continue
        return number

    return default


def optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_float(value: Any) -> Any:
    if value is None:
        return None
    coerced = optional_float(value)
    return coerced if coerced is not None else value


def maybe_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        try:
            if value is None:
                continue
            number = float(value)
            if math.isnan(number):
                continue
            return number
        except (TypeError, ValueError):
            continue
    return float(default)


def round_float(value: Any, decimals: int = 2, default: float = 0.0) -> float:
    numeric = optional_float(value)
    if numeric is None:
        return float(default)
    try:
        return round(numeric, decimals)
    except Exception:
        return float(default)


def round_price(value: Any) -> float:
    price = optional_float(value)
    if price is None:
        return 0.0
    abs_price = abs(price)
    if abs_price >= 1000:
        decimals = 1
    elif abs_price >= 100:
        decimals = 2
    elif abs_price >= 10:
        decimals = 3
    elif abs_price >= 1:
        decimals = 4
    elif abs_price >= 0.1:
        decimals = 5
    elif abs_price >= 0.01:
        decimals = 6
    else:
        decimals = 7
    try:
        return round(price, decimals)
    except Exception:
        return 0.0


def _round_series(
    values: Iterable[Any],
    *,
    count: int,
    rounder: Callable[[Optional[float]], Optional[float]],
) -> List[float]:
    collected = list(values)
    if not collected:
        return []
    result: List[float] = []
    for value in collected[-count:]:
        numeric = optional_float(value)
        if numeric is None or math.isnan(numeric):
            continue
        rounded = rounder(numeric)
        if rounded is None:
            continue
        result.append(rounded)
    return result


def round_numeric_series(values: Iterable[Any], *, count: int, decimals: int) -> List[float]:
    return _round_series(
        values,
        count=count,
        rounder=lambda numeric: round(numeric, decimals) if not math.isnan(numeric) else None,
    )


def round_price_series(values: Iterable[Any], *, count: int) -> List[float]:
    return _round_series(values, count=count, rounder=lambda numeric: round_price(numeric))


def safe_load(source: Union[str, Path, IO[str]], *, default: Any = None) -> Any:
    """Load JSON from *source*; return *default* when parsing fails."""
    try:
        if isinstance(source, Path):
            return json.loads(source.read_text())
        if hasattr(source, "read"):
            return json.load(source)  # type: ignore[arg-type]
        return json.loads(str(source))
    except Exception:
        return default


def safe_dump(
    payload: Any,
    target: Union[str, Path, IO[str]],
    *,
    ensure_ascii: bool = False,
    indent: Optional[int] = 2,
) -> bool:
    """Serialize *payload* to *target* safely; return False if writing fails."""
    try:
        content = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
        if isinstance(target, Path):
            target.write_text(content)
        elif hasattr(target, "write"):
            target.write(content)  # type: ignore[arg-type]
        else:
            Path(str(target)).write_text(content)
        return True
    except Exception:
        return False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(ts: float, *, tz_offset_hours: int = 8, fmt: str = "%m-%d %H:%M:%S") -> str:
    tz = timezone(timedelta(hours=tz_offset_hours))
    return datetime.fromtimestamp(ts, tz=tz).strftime(fmt)


def canonical_market_symbol(symbol: str) -> str:
    primary = str(symbol).split(":")[0]
    return primary.replace("/", "").upper()


def find_market_symbol(symbol: str, markets: Mapping[str, Any]) -> Optional[str]:
    target = canonical_market_symbol(symbol)
    for market_symbol in markets.keys():
        if canonical_market_symbol(market_symbol) == target:
            return market_symbol
    return None


def to_ccxt_symbol(symbol: str, markets: Optional[Mapping[str, Any]] = None) -> str:
    if not symbol:
        return ""
    normalized = symbol.upper()
    normalized_nocolon = normalized.split(":")[0] if ":" in normalized else normalized
    if markets:
        match = find_market_symbol(normalized, markets)
        if match:
            return match
        match = find_market_symbol(normalized_nocolon, markets)
        if match:
            return match
    if "/" in normalized:
        candidate = normalized
    elif normalized.endswith("USDT"):
        candidate = f"{normalized[:-4]}/USDT"
    else:
        candidate = normalized
    if markets:
        match = find_market_symbol(candidate, markets)
        if match:
            return match
        if candidate.endswith("/USDT"):
            base = candidate.split("/")[0]
            base_clean = base.replace("/", "")
            for multiplier in ("1000", "10000", "100000", "1000000"):
                alt = f"{multiplier}{base_clean}/USDT"
                match = find_market_symbol(alt, markets)
                if match:
                    return match
    return candidate


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("/", "")
    if symbol.endswith("USDT"):
        return symbol
    if "USDT" not in symbol:
        return f"{symbol}USDT"
    return symbol


def normalize_decision_symbol(decision: Dict[str, Any]) -> str:
    sym = decision.get("symbol")
    if not isinstance(sym, str):
        return ""
    return sym.upper().replace("/", "")


def timeframe_to_seconds(timeframe: str) -> int:
    try:
        numeric = int(timeframe[:-1])
        unit = timeframe[-1].lower()
    except (ValueError, IndexError, TypeError):
        return 0
    factor = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }.get(unit)
    if factor is None:
        return 0
    return numeric * factor


def trim_to_closed_bars(
    bars: Sequence[Sequence[Any]],
    timeframe_seconds: int,
) -> Tuple[List[List[Any]], Optional[int]]:
    if not bars:
        return [], None
    series: List[List[Any]] = [list(bar) for bar in bars if bar is not None]
    if not series:
        return [], None
    last_open = ensure_float(series[-1][0]) / 1000.0
    now = utc_now().timestamp()
    if timeframe_seconds > 0 and now < last_open + timeframe_seconds:
        series = series[:-1]
    if not series:
        return [], None
    last_closed_ts = int(series[-1][0])
    return series, last_closed_ts


def is_close_to_zero(value: float, *, abs_tol: float = 1e-12) -> bool:
    return math.isclose(value, 0.0, abs_tol=abs_tol)


__all__ = [
    "ensure_float",
    "optional_float",
    "coerce_optional_float",
    "maybe_float",
    "round_float",
    "round_price",
    "round_numeric_series",
    "round_price_series",
    "safe_dump",
    "safe_load",
    "utc_now",
    "format_ts",
    "canonical_market_symbol",
    "find_market_symbol",
    "to_ccxt_symbol",
    "normalize_symbol",
    "normalize_decision_symbol",
    "timeframe_to_seconds",
    "trim_to_closed_bars",
    "is_close_to_zero",
    "first_float",
]
