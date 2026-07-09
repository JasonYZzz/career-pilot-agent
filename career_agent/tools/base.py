from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from career_agent.runtime.run_state import RunState, ToolResult


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    risk_level: str
    timeout_ms: int = 3000


class Tool(Protocol):
    meta: ToolMeta

    def run(self, args: dict[str, Any], state: RunState) -> ToolResult:
        ...

