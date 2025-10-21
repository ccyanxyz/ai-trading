"""Built-in market adapters for the AI framework."""

from .base import BaseMarketAdapter
from .binance_perp import BinancePerpetualAdapter
from .binance_spot import BinanceSpotAdapter
from .tiger_futures import TigerFuturesAdapter
from .tiger_stocks import TigerEquityAdapter
from .yfinance_stocks import YFinanceEquityAdapter

__all__ = [
    "BaseMarketAdapter",
    "BinancePerpetualAdapter",
    "BinanceSpotAdapter",
    "TigerFuturesAdapter",
    "TigerEquityAdapter",
    "YFinanceEquityAdapter",
]
