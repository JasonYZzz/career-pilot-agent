import json
import time

from career_agent.runtime.run_state import RunState, ToolResult
from career_agent.tools.base import ToolMeta


class CreateReminderTool:
    meta = ToolMeta(
        name="create_reminder",
        description="Create reminder draft unless confirmed.",
        risk_level="high",
        timeout_ms=2000,
    )

    def run(self, args: dict[str, object], state: RunState) -> ToolResult:
        started = time.perf_counter()
        outputs = state.workspace / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        trusted_confirmation = args.get("_user_confirmed") is True
        item = {
            "title": str(args["title"]),
            "date": str(args["date"]),
            "note": str(args.get("note", "")),
            "status": "confirmed" if trusted_confirmation else "draft_requires_confirmation",
        }
        path = outputs / "reminder_plan.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(item)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        state.boundary_events.append(
            {
                "type": "reminder_requires_confirmation",
                "title": item["title"],
                "path": "outputs/reminder_plan.json",
            }
        )
        return ToolResult(
            ok=True,
            content="reminder draft written; confirmation required",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            metadata={"path": "outputs/reminder_plan.json"},
        )
