"""DAG runner — topological sort + execute callable per node.

Examples:
    >>> from naviertwin.utils.workflow.dag_runner import DAGRunner
    >>> r = DAGRunner()
    >>> r.add('a', lambda inputs: 1)
    >>> r.add('b', lambda inputs: inputs['a'] + 1, deps=['a'])
    >>> r.run()['b']
    2
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DAGRunner:
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[dict], Any]] = {}
        self.deps: dict[str, list[str]] = {}

    def add(self, name: str, fn: Callable[[dict], Any], *,
            deps: list[str] | None = None) -> None:
        self.nodes[name] = fn
        self.deps[name] = list(deps or [])

    def topo(self) -> list[str]:
        order: list[str] = []
        visited: set[str] = set()

        def dfs(n: str) -> None:
            if n in visited:
                return
            for d in self.deps.get(n, []):
                dfs(d)
            visited.add(n)
            order.append(n)

        for n in self.nodes:
            dfs(n)
        return order

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for n in self.topo():
            results[n] = self.nodes[n](results)
        return results


__all__ = ["DAGRunner"]
