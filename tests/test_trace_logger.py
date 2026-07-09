import json
from pathlib import Path

from career_agent.runtime.run_state import RunState
from career_agent.runtime.trace_logger import TraceLogger


def test_trace_logger_exports_summary(tmp_path: Path) -> None:
    state = RunState(run_id="run_test", task="生成职业规划", workspace=tmp_path, max_steps=12)
    logger = TraceLogger(
        run_id=state.run_id,
        task=state.task,
        workspace=tmp_path,
        llm_provider="mock",
        llm_protocol="mock",
        llm_model="mock-career-planner",
    )
    logger.add_span("model_call", name="planner", estimated_input_tokens=120, elapsed_ms=10)
    logger.add_span("tool_call", name="list_dir", ok=True, elapsed_ms=3)
    logger.add_span("skill_load", name="career_assessment", estimated_tokens=300, elapsed_ms=2)
    trace_path = tmp_path / "trace.json"
    logger.export(trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_test"
    assert data["summary"]["model_calls"] == 1
    assert data["summary"]["tool_calls"] == 1
    assert data["summary"]["skill_loads"] == 1
    assert data["spans"][0]["span_id"] == "span_001"

