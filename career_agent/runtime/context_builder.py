from __future__ import annotations

import json

from career_agent.runtime.run_state import RunState


class ContextBuilder:
    def build(self, state: RunState) -> str:
        payload = {
            "task": state.task,
            "step": state.step,
            "max_steps": state.max_steps,
            "todos": state.todos,
            "loaded_skills": list(state.loaded_skills),
            "compressed_summary": state.compressed_summary,
            "recent_tool_results": self._recent_tool_results(state),
            "boundary_events": state.boundary_events[-5:],
            "available_tools": [
                "list_dir(path)",
                "read_file(path,max_chars)",
                "write_file(path under outputs/,content,mode)",
                "todo_update(items)",
                "get_time()",
                "create_reminder(title,date,note,confirmed=false)",
                "restricted_shell(command,timeout_ms)",
            ],
            "available_skills": [
                "career_assessment",
                "role_matching",
                "skill_gap_analysis",
                "action_plan",
                "report_writer",
            ],
            "decision_rules": [
                "Return exactly one JSON object.",
                "Never use absolute paths or .. path segments.",
                "write_file can only write outputs/*.",
                "Reminder creation must be draft-only unless user confirmation is explicit.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _recent_tool_results(self, state: RunState) -> list[dict[str, object]]:
        if state.compressed_summary:
            source = state.tool_results[state.last_compression_tool_result_count :]
        else:
            source = state.tool_results[-5:]
        return [
            {
                "tool": item.get("tool"),
                "path": item.get("path"),
                "content": str(item.get("content", ""))[:1200],
                "flags": item.get("flags", []),
                "truncated": item.get("truncated", False),
            }
            for item in source[-5:]
        ]
