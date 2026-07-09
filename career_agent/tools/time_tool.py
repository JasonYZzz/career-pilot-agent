from datetime import datetime
import time

from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import ToolMeta


class GetTimeTool:
    meta = ToolMeta(name="get_time", description="Get local time.", risk_level="low")

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        _ = args
        return ToolResult(
            ok=True,
            content=datetime.now().astimezone().isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

