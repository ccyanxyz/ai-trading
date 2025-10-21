"""Utility helpers for rendering indicator-rich candlestick charts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, List, Any

import numpy as np
import pandas as pd
import pytz


CHART_TZ = pytz.timezone("Asia/Shanghai")


@dataclass
class ChartConfig:
    symbol: str
    interval: str
    output_path: Path
    indicators: Optional[Sequence[str]] = None


class ChartService:
    """Render and store candlestick charts with optional indicator overlays."""

    DEFAULT_INDICATORS: Sequence[str] = ("ema20", "ema60", "ema120", "rsi", "macd")

    @staticmethod
    def render(df: pd.DataFrame, config: ChartConfig) -> Path:
        if df.empty:
            raise ValueError("Empty dataframe for chart rendering")

        try:  # pragma: no cover - heavy optional dependency
            import mplfinance as mpf
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mplfinance and matplotlib are required for chart rendering"
            ) from exc

        close = df["close"]

        if config.indicators is None:
            indicator_names: List[str] = list(ChartService.DEFAULT_INDICATORS)
        else:
            indicator_names = [str(name).lower() for name in config.indicators if name]

        has_ema20 = "ema20" in indicator_names
        has_ema60 = "ema60" in indicator_names
        has_ema120 = "ema120" in indicator_names
        has_rsi = "rsi" in indicator_names
        has_macd = "macd" in indicator_names

        ema20 = ema60 = ema120 = None
        if has_ema20:
            ema20 = close.ewm(span=20, adjust=False).mean()
        if has_ema60:
            ema60 = close.ewm(span=60, adjust=False).mean()
        if has_ema120:
            ema120 = close.ewm(span=120, adjust=False).mean()

        rsi = None
        if has_rsi:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta.clip(upper=0))
            avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(0)

        macd_line = signal_line = macd_hist = None
        if has_macd:
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line

        mc = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            wick="inherit",
            edge="inherit",
            volume="in",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#0e1114",
            edgecolor="#1b1f24",
            gridcolor="#22262b",
            figcolor="#0e1114",
            rc={"font.size": 8},
        )

        add_plots: List[Any] = []
        panel_ratios: List[int] = [8, 2]
        next_panel = 2

        if has_ema20 and ema20 is not None:
            add_plots.append(mpf.make_addplot(ema20, panel=0, color="#e0e0e0", width=1.0))
        if has_ema60 and ema60 is not None:
            add_plots.append(mpf.make_addplot(ema60, panel=0, color="#9fa8da", width=1.0))
        if has_ema120 and ema120 is not None:
            add_plots.append(mpf.make_addplot(ema120, panel=0, color="#f6d743", width=1.0))

        if has_rsi and rsi is not None:
            panel_index = next_panel
            next_panel += 1
            panel_ratios.append(2)
            add_plots.extend(
                [
                    mpf.make_addplot(rsi, panel=panel_index, color="#00bcd4", width=1.0, ylabel="RSI"),
                    mpf.make_addplot(
                        pd.Series(70, index=df.index),
                        panel=panel_index,
                        color="#ff8a65",
                        linestyle="--",
                        width=0.8,
                    ),
                    mpf.make_addplot(
                        pd.Series(30, index=df.index),
                        panel=panel_index,
                        color="#81c784",
                        linestyle="--",
                        width=0.8,
                    ),
                ]
            )

        if has_macd and macd_line is not None and signal_line is not None and macd_hist is not None:
            panel_index = next_panel
            next_panel += 1
            panel_ratios.append(3)
            add_plots.extend(
                [
                    mpf.make_addplot(macd_line, panel=panel_index, color="#42a5f5", width=1.0, ylabel="MACD"),
                    mpf.make_addplot(signal_line, panel=panel_index, color="#ffca28", width=1.0),
                    mpf.make_addplot(
                        macd_hist,
                        panel=panel_index,
                        type="bar",
                        color=["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist],
                        alpha=0.6,
                    ),
                ]
            )

        last_close = float(close.iloc[-1])
        title_time = df.index[-1].strftime("%Y-%m-%d %H:%M")
        title = f"{config.symbol} — {config.interval.upper()} — last: {title_time}"

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            addplot=add_plots,
            panel_ratios=tuple(panel_ratios),
            datetime_format="%m-%d %H:%M",
            tight_layout=True,
            figratio=(16, 9),
            figscale=1.2,
            returnfig=True,
        )

        for ax in axes:
            try:
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position("right")
            except Exception:  # pragma: no cover - fallback for axes without standard API
                continue
            if "left" in ax.spines:
                ax.spines["left"].set_visible(False)
            if "right" in ax.spines:
                ax.spines["right"].set_visible(True)

        price_ax = axes[0]
        legend_handles = [
            price_ax.plot([], [], color="#ffffff", label=f"Price: {format_price(last_close)}")[0]
        ]
        if has_ema20 and ema20 is not None:
            legend_handles.append(
                price_ax.plot([], [], color="#e0e0e0", label=f"EMA20: {format_price(ema20.iloc[-1])}")[0]
            )
        if has_ema60 and ema60 is not None:
            legend_handles.append(
                price_ax.plot([], [], color="#9fa8da", label=f"EMA60: {format_price(ema60.iloc[-1])}")[0]
            )
        if has_ema120 and ema120 is not None:
            legend_handles.append(
                price_ax.plot([], [], color="#f6d743", label=f"EMA120: {format_price(ema120.iloc[-1])}")[0]
            )
        price_ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False, labelcolor="white")

        fig.savefig(config.output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return config.output_path


def klines_to_dataframe(klines: Iterable[Iterable[float]], *, limit: int) -> pd.DataFrame:
    """Convert raw kline arrays to a pandas DataFrame with timezone-aware index."""

    rows = list(klines)[-limit:]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(CHART_TZ)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.set_index("open_time")
    df.index.name = "Time"
    return df


def format_price(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        decimals = 2
    elif abs_value >= 10:
        decimals = 3
    elif abs_value >= 1:
        decimals = 4
    elif abs_value >= 0.1:
        decimals = 5
    elif abs_value >= 0.01:
        decimals = 6
    elif abs_value >= 0.001:
        decimals = 7
    else:
        decimals = 8
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".") or "0"
