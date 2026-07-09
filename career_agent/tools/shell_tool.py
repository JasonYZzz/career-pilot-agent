import subprocess
import time

from career_agent.runtime.boundary_guard import BoundaryGuard
from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import ToolMeta


class RestrictedShellTool:
    meta = ToolMeta(
        name="restricted_shell",
        description="Run allowlisted shell commands inside workspace.",
        risk_level="high",
        timeout_ms=5000,
    )

    def __init__(self, guard: BoundaryGuard) -> None:
        self.guard = guard

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        command = str(args["command"])
        ok, reason = self.guard.validate_shell_command(command)
        if not ok:
            return ToolResult(
                ok=False,
                content="",
                error=reason,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        try:
            completed = subprocess.run(
                command.split(),
                cwd=state.workspace,
                capture_output=True,
                text=True,
                timeout=int(str(args.get("timeout_ms", 3000))) / 1000,
                check=False,
            )
            return ToolResult(
                ok=completed.returncode == 0,
                content=completed.stdout[:6000],
                error=completed.stderr[:2000] or None,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                content="",
                error="tool_timeout",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
