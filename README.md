# AI-Driven Trading Framework

> ⚠️ **Experimental Project**: This is research code from personal trading experiments. Not production-ready, not profitable. Use at your own risk.

## Overview

A modular AI-powered trading framework supporting multiple exchanges and strategies. The system uses LLM-based decision-making with risk management and position tracking.

## Architecture

```
ai-trading/
├── core/           # Core abstractions and interfaces
├── adapters/       # Exchange connectors (Binance, Tiger, yFinance)
├── engine/         # AI strategy execution engine
├── strategies/     # Strategy implementations
├── services/       # Support services (Telegram, charts, AI client)
├── execution/      # Order execution and guards
├── exit_engine/    # Exit rules and position management
└── filters/        # Pre-trade filters
```

## Modules

### Core (`core/`)
- **interfaces.py**: Base contracts for adapters, formatters, and strategies
- **context.py**: Trading context and state management
- **position.py**: Position tracking and execution history
- **defaults.py**: Default implementations and utilities

### Adapters (`adapters/`)
Exchange-specific connectors implementing unified interface:
- **binance_perp.py**: Binance USDT-M perpetual futures
- **binance_spot.py**: Binance spot trading
- **tiger_stocks.py**: Tiger Brokers US stocks
- **tiger_futures.py**: Tiger Brokers futures
- **yfinance_stocks.py**: YFinance data adapter (read-only)

### Engine (`engine/`)
- **strategy.py**: Base AI strategy with LLM integration
- **pipeline.py**: Decision-making pipeline orchestration

### Strategies (`strategies/`)
- **binance_perp.py**: AI-driven perpetual futures strategy with technical analysis
- **common/decision_utils.py**: Decision payload normalization

### Services (`services/`)
- **ai_client.py**: LLM API client (OpenAI/Anthropic compatible)
- **telegram.py**: Telegram bot for monitoring and manual control
- **chart.py**: Chart generation with indicators
- **position_manager.py**: Multi-symbol position state management
- **scheduler.py**: Strategy execution scheduling
- **state.py**: Persistent state storage

### Execution (`execution/`)
- **guards.py**: Pre-execution risk checks and validations

### Exit Engine (`exit_engine/`)
- **engine.py**: Exit decision engine
- **rules.py**: Exit rule implementations (stop-loss, take-profit, trailing)
- **models.py**: Exit-related data models

### Filters (`filters/`)
- **pre_filters.py**: Pre-trade filtering (volatility, liquidity, etc.)

## Key Features

- 🤖 **LLM-Based Decisions**: Leverages Claude/GPT for market analysis
- 📊 **Technical Analysis**: Built-in indicators (EMA, RSI, MACD, ATR)
- 📈 **Chart Generation**: Automated chart creation for visual analysis
- 🎯 **Risk Management**: Position sizing, stop-loss, take-profit
- 🔔 **Telegram Integration**: Real-time alerts and manual control
- 💾 **State Persistence**: Position tracking across restarts
- 🔌 **Multi-Exchange**: Unified interface for different brokers

## Configuration

Set required environment variables:

```bash
# Exchange credentials
export BINANCE_USDM_API_KEY="your_key"
export BINANCE_USDM_API_SECRET="your_secret"

# Telegram (optional)
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export TELEGRAM_ADMIN_IDS="comma,separated,ids"

# AI provider (OpenAI or Anthropic)
export OPENAI_API_KEY="your_key"
# or
export ANTHROPIC_API_KEY="your_key"
```

## Usage Example

```python
from ai_trading.strategies.binance_perp import AIBinancePerpStrategy

config = {
    "exchange": "binance",
    "api_key_env": "BINANCE_USDM_API_KEY",
    "api_secret_env": "BINANCE_USDM_API_SECRET",
    "default_symbols": ["BTCUSDT", "ETHUSDT"],
    "ai": {
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "max_tokens": 4000
    },
    "constraints": {
        "long_only": False,
        "risk_preference": "medium"
    }
}

strategy = AIBinancePerpStrategy(config)
await strategy.run()
```

## Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

- This code is experimental and unaudited
- Trading carries significant financial risk
- No guarantees of profitability
- Use at your own risk
- Author assumes no liability for losses

## Contributing

This is a personal research project. Feel free to fork and experiment, but no support is provided.
