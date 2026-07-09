from __future__ import annotations

from career_agent.runtime.boundary_guard import BoundaryGuard
from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import Tool
from career_agent.tools.file_tools import ListDirTool, ReadFileTool, WriteFileTool
from career_agent.tools.reminder_tool import CreateReminderTool
from career_agent.tools.shell_tool import RestrictedShellTool
from career_agent.tools.time_tool import GetTimeTool
from career_agent.tools.todo_tool import TodoUpdateTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.meta.name] = tool

    def run(self, name: str, args: dict[str, object], state: RunState) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ok=False, content="", error=f"unknown tool: {name}")
        return self._tools[name].run(args, state)


def build_default_tool_registry(guard: BoundaryGuard) -> ToolRegistry:
    registry = ToolRegistry()
    tools: tuple[Tool, ...] = (
        ListDirTool(guard),
        ReadFileTool(guard),
        WriteFileTool(guard),
        TodoUpdateTool(),
        GetTimeTool(),
        CreateReminderTool(),
        RestrictedShellTool(guard),
    )
    for tool in tools:
        registry.register(tool)
    return registry
