"""Coordinator that evaluates exit rules and emits decisions."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .models import ExitContext, ExitDecision
from .rules import ExitRule


class RuleExitEngine:
    """Executes registered exit rules in order."""

    def __init__(self, rules: Sequence[ExitRule]) -> None:
        self._rules: List[ExitRule] = list(rules)

    def evaluate(self, ctx: ExitContext) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        for rule in self._rules:
            decision = rule.evaluate(ctx)
            if decision:
                decisions.append(decision)
        return decisions

    def extend(self, rules: Iterable[ExitRule]) -> None:
        self._rules.extend(rules)

    def clear(self) -> None:
        self._rules.clear()


__all__ = ["RuleExitEngine"]
