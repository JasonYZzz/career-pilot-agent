import time

from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import ToolMeta


class TodoUpdateTool:
    meta = ToolMeta(name="todo_update", description="Update todo state.", risk_level="low")

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        items = args.get("items", [])
        state.todos = list(items) if isinstance(items, list) else []
        return ToolResult(
            ok=True,
            content=f"updated {len(state.todos)} todo items",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

