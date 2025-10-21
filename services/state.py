"""File-backed persistence helpers for AI trading strategies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..utils import format_ts, utc_now, safe_dump, safe_load


class StrategyStateStore:
    """Persists strategy-scoped data such as watchlists and trade plans."""

    def __init__(
        self,
        base_dir: Path,
        strategy_key: str,
        *,
        normalize_symbol: Callable[[str], str],
        logger: logging.Logger,
    ) -> None:
        self.strategy_key = strategy_key
        self._normalize_symbol = normalize_symbol
        self._logger = logger

        self._storage_dir = base_dir / strategy_key
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._watchlist_path = self._storage_dir / "watchlist.json"
        self._plans_path = self._storage_dir / "trade_plans.json"
        self._positions_path = self._storage_dir / "positions.json"
        self._archived_positions_dir = self._storage_dir / "archived_positions"
        self._processed_path = self._storage_dir / "last_processed.json"
        self._decisions_path = self._storage_dir / "decisions.json"
        self._execution_log_path = self._storage_dir / "executions.log"

    # ------------------------------------------------------------------
    # Basic filesystem accessors
    # ------------------------------------------------------------------

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def decisions_path(self) -> Path:
        return self._decisions_path

    @property
    def execution_log_path(self) -> Path:
        return self._execution_log_path

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------

    def load_watchlist(self) -> List[str]:
        if not self._watchlist_path.exists():
            return []
        data = safe_load(self._watchlist_path, default=None)
        if data is None:
            self._logger.warning("Failed to load watchlist: %s", self._watchlist_path)
            return []
        if isinstance(data, list):
            return [self._normalize_symbol(str(sym)) for sym in data if sym]
        return []

    def save_watchlist(self, symbols: Sequence[str]) -> None:
        if not safe_dump(list(symbols), self._watchlist_path, ensure_ascii=False):
            self._logger.warning("Failed to persist watchlist: %s", self._watchlist_path)

    # ------------------------------------------------------------------
    # Trade plan persistence
    # ------------------------------------------------------------------

    def load_trade_plans(self) -> Dict[str, Dict[str, Any]]:
        if not self._plans_path.exists():
            return {}
        data = safe_load(self._plans_path, default=None)
        if data is None:
            self._logger.warning("Failed to load trade plans: %s", self._plans_path)
            return {}
        if not isinstance(data, dict):
            return {}
        results: Dict[str, Dict[str, Any]] = {}
        for sym, payload in data.items():
            if isinstance(payload, dict):
                normalized = self._normalize_symbol(str(sym))
                results[normalized] = payload
        return results

    def save_trade_plans(self, plans: Dict[str, Dict[str, Any]]) -> None:
        if not safe_dump(plans, self._plans_path, ensure_ascii=False):
            self._logger.warning("Failed to persist trade plans: %s", self._plans_path)

    # ------------------------------------------------------------------
    # Position persistence
    # ------------------------------------------------------------------

    def load_positions(self) -> Dict[str, Dict[str, Any]]:
        if not self._positions_path.exists():
            return {}
        data = safe_load(self._positions_path, default=None)
        if data is None:
            self._logger.warning("Failed to load positions: %s", self._positions_path)
            return {}
        if not isinstance(data, dict):
            return {}
        results: Dict[str, Dict[str, Any]] = {}
        for sym, payload in data.items():
            if isinstance(payload, dict):
                normalized = self._normalize_symbol(str(sym))
                results[normalized] = payload
        return results

    def save_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        if not safe_dump(positions, self._positions_path, ensure_ascii=False):
            self._logger.warning("Failed to persist positions: %s", self._positions_path)

    def archive_position(self, symbol: str, position: Dict[str, Any]) -> None:
        try:
            self._archived_positions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._logger.exception(
                "Failed to ensure archived positions directory exists: %s",
                self._archived_positions_dir,
            )
            return

        timestamp = position.get("timestamp") or format_ts(utc_now().timestamp())
        filename = f"{symbol}_{timestamp}.json"
        filepath = self._archived_positions_dir / filename
        if not safe_dump(position, filepath, ensure_ascii=False):
            self._logger.warning("Failed to persist archived position: %s", filepath)

    # ------------------------------------------------------------------
    # Last processed timestamps
    # ------------------------------------------------------------------

    def load_last_processed(self) -> Dict[str, int]:
        if not self._processed_path.exists():
            return {}
        data = safe_load(self._processed_path, default=None)
        if data is None:
            self._logger.warning("Failed to load last processed timestamps: %s", self._processed_path)
            return {}
        if not isinstance(data, dict):
            return {}
        results: Dict[str, int] = {}
        for sym, value in data.items():
            normalized = self._normalize_symbol(str(sym))
            try:
                results[normalized] = int(value)
            except (TypeError, ValueError):
                continue
        return results

    def save_last_processed(self, payload: Dict[str, int]) -> None:
        if not safe_dump(payload, self._processed_path, ensure_ascii=False):
            self._logger.warning("Failed to persist last processed timestamps: %s", self._processed_path)

    # ------------------------------------------------------------------
    # Decision and execution history helpers
    # ------------------------------------------------------------------

    def persist_decision(
        self,
        timestamp_iso: str,
        response_ids: Sequence[str],
        account_payload: Any,
        decisions: Sequence[Dict[str, Any]],
    ) -> None:
        record = {
            "timestamp": timestamp_iso,
            "response_ids": list(response_ids),
            "account": self._json_safe(account_payload),
            "decisions": self._json_safe(list(decisions)),
        }
        existing: List[Dict[str, Any]] = []
        if self._decisions_path.exists():
            try:
                existing_data = json.loads(self._decisions_path.read_text())
                if isinstance(existing_data, list):
                    existing = existing_data
            except Exception:
                self._logger.exception(
                    "Failed to load existing decisions file: %s",
                    self._decisions_path,
                )
        existing.append(record)
        try:
            content = json.dumps(existing, ensure_ascii=False, indent=2)
            self._decisions_path.write_text(content)
        except Exception:
            self._logger.exception("Failed to persist decision file: %s", self._decisions_path)

    def append_execution_log(self, record: Dict[str, Any]) -> None:
        try:
            with self._execution_log_path.open("a", encoding="utf-8") as handle:
                safe_record = self._json_safe(record)
                handle.write(json.dumps(safe_record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            self._logger.exception("Failed to write execution log: %s", self._execution_log_path)

    def archive_trade(self, symbol: str, entry: Dict[str, Any]) -> None:
        history_dir = self._storage_dir / "history_trades"
        try:
            history_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._logger.exception(
                "Failed to ensure history_trades directory exists: %s",
                history_dir,
            )
            return

        trade_plan_id = entry.get("trade_plan_id") or format_ts(utc_now().timestamp())
        filename = f"{symbol}_{trade_plan_id}.json"
        filepath = history_dir / filename
        try:
            content = json.dumps(entry, ensure_ascii=False, indent=2)
            filepath.write_text(content)
        except Exception:
            self._logger.exception("Failed to persist archived trade: %s", filepath)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return value


__all__ = ["StrategyStateStore"]
