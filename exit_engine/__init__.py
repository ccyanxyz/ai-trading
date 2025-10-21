"""Rule-driven exit engine for deterministic TP/SL management."""

from .models import ExitContext, ExitDecision
from .rules import (
    ExitRule,
    StopLossExitRule,
    TargetExitRule,
    BreakEvenStopRule,
    TrailingExitRule,
)
from .engine import RuleExitEngine

__all__ = [
    "ExitContext",
    "ExitDecision",
    "ExitRule",
    "StopLossExitRule",
    "TargetExitRule",
    "BreakEvenStopRule",
    "TrailingExitRule",
    "RuleExitEngine",
]
