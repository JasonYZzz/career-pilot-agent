import json
from pathlib import Path


def test_example_trace_contains_required_spans() -> None:
    path = Path("examples/trace_with_compression.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    span_types = {span["type"] for span in data["spans"]}
    tool_names = {span.get("name") for span in data["spans"] if span["type"] == "tool_call"}
    boundary_events = {
        span.get("event_type") for span in data["spans"] if span["type"] == "boundary_event"
    }

    assert data["termination_reason"] == "final_answer"
    assert "model_call" in span_types
    assert "tool_call" in span_types
    assert "skill_load" in span_types
    assert "token_budget" in span_types
    assert "compression" in span_types
    assert "boundary_event" in span_types
    assert {"read_file", "write_file", "create_reminder"}.issubset(tool_names)
    assert "prompt_injection_detected" in boundary_events
    assert "reminder_requires_confirmation" in boundary_events
