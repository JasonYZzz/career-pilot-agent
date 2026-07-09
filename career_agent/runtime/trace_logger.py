from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from career_agent.runtime.run_state import RunState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class TraceLogger:
    def __init__(
        self,
        run_id: str,
        task: str,
        workspace: Path,
        llm_provider: str,
        llm_protocol: str,
        llm_model: str,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.workspace = str(workspace)
        self.started_at = utc_now_iso()
        self.llm_provider = llm_provider
        self.llm_protocol = llm_protocol
        self.llm_model = llm_model
        self.spans: list[dict[str, Any]] = []

    def add_span(self, span_type: str, **payload: Any) -> dict[str, Any]:
        span = {
            "type": span_type,
            "span_id": f"span_{len(self.spans) + 1:03d}",
            "timestamp": utc_now_iso(),
            **payload,
        }
        self.spans.append(span)
        return span

    def export(self, path: Path, state: RunState) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "task": self.task,
            "workspace": self.workspace,
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "status": "success" if state.done and state.termination_reason != "error" else "partial",
            "model": {
                "provider": self.llm_provider,
                "api_mode": self.llm_protocol,
                "name": self.llm_model,
            },
            "config": {"max_steps": state.max_steps},
            "termination_reason": state.termination_reason,
            "spans": self.spans,
            "summary": self._summary(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _summary(self) -> dict[str, Any]:
        return {
            "model_calls": sum(1 for span in self.spans if span["type"] == "model_call"),
            "tool_calls": sum(1 for span in self.spans if span["type"] == "tool_call"),
            "skill_loads": sum(1 for span in self.spans if span["type"] == "skill_load"),
            "compressions": sum(1 for span in self.spans if span["type"] == "compression"),
            "boundary_events": sum(1 for span in self.spans if span["type"] == "boundary_event"),
            "total_estimated_tokens": sum(
                int(span.get("estimated_input_tokens", 0))
                + int(span.get("estimated_output_tokens", 0))
                for span in self.spans
            ),
        }

