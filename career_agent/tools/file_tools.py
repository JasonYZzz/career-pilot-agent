from __future__ import annotations

import time

from career_agent.runtime.boundary_guard import BoundaryGuard
from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import ToolMeta


class ListDirTool:
    meta = ToolMeta(
        name="list_dir",
        description="List files inside workspace.",
        risk_level="low",
        timeout_ms=2000,
    )

    def __init__(self, guard: BoundaryGuard) -> None:
        self.guard = guard

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        try:
            path = self.guard.ensure_workspace_path(str(args.get("path", ".")), state.workspace)
            entries = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir())
            return ToolResult(
                ok=True,
                content="\n".join(entries),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                metadata={"path": str(args.get("path", "."))},
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=str(exc),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )


class ReadFileTool:
    meta = ToolMeta(
        name="read_file",
        description="Read a text file inside workspace.",
        risk_level="medium",
        timeout_ms=3000,
    )

    def __init__(self, guard: BoundaryGuard) -> None:
        self.guard = guard

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        try:
            path_arg = str(args["path"])
            max_chars = int(str(args.get("max_chars", 6000)))
            path = self.guard.ensure_workspace_path(path_arg, state.workspace)
            text = path.read_text(encoding="utf-8")
            flags = self.guard.scan_untrusted_text(text)
            truncated = len(text) > max_chars
            content = text[:max_chars]
            result = ToolResult(
                ok=True,
                content=content,
                truncated=truncated,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                metadata={"path": path_arg, "flags": flags},
            )
            state.tool_results.append(
                {
                    "tool": "read_file",
                    "path": path_arg,
                    "content": self.guard.mask_privacy(content),
                    "truncated": truncated,
                    "flags": flags,
                }
            )
            return result
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=str(exc),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )


class WriteFileTool:
    meta = ToolMeta(
        name="write_file",
        description="Write a text file under workspace/outputs.",
        risk_level="medium",
        timeout_ms=3000,
    )

    def __init__(self, guard: BoundaryGuard) -> None:
        self.guard = guard

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        try:
            path_arg = str(args["path"])
            content = str(args.get("content", ""))
            mode = str(args.get("mode", "overwrite"))
            path = self.guard.ensure_output_path(path_arg, state.workspace)
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(content)
            else:
                path.write_text(content, encoding="utf-8")
            return ToolResult(
                ok=True,
                content=f"wrote {path_arg}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                metadata={"path": path_arg},
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=str(exc),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
