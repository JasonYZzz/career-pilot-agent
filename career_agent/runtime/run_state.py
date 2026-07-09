from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


DecisionType = Literal[
    "call_tool",
    "load_skill",
    "update_todo",
    "compress_context",
    "final_answer",
    "ask_clarification",
]


@dataclass
class AgentDecision:
    decision: DecisionType
    reason: str
    thought_summary: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    skill_name: str | None = None
    todo_update: list[dict[str, Any]] | None = None
    final_answer: str | None = None
    expected_observation: str | None = None


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None
    truncated: bool = False
    elapsed_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    run_id: str
    task: str
    workspace: Path
    step: int = 0
    max_steps: int = 12
    messages: list[dict[str, Any]] = field(default_factory=list)
    loaded_skills: dict[str, str] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    compressed_summary: dict[str, Any] | None = None
    last_compression_tool_result_count: int = 0
    boundary_events: list[dict[str, Any]] = field(default_factory=list)
    repeated_failures: dict[str, int] = field(default_factory=dict)
    done: bool = False
    final_answer: str | None = None
    termination_reason: str | None = None
