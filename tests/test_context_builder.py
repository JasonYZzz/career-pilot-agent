from pathlib import Path

from career_agent.runtime.context_builder import ContextBuilder
from career_agent.runtime.run_state import RunState


def test_context_includes_post_compression_tool_results(tmp_path: Path) -> None:
    state = RunState(run_id="run_test", task="生成报告", workspace=tmp_path)
    state.compressed_summary = {"task_goal": "生成报告"}
    state.last_compression_tool_result_count = 1
    state.tool_results = [
        {"tool": "read_file", "path": "data/old.md", "content": "old", "flags": []},
        {
            "tool": "get_time",
            "path": "",
            "content": "2026-07-07T20:00:00+08:00",
            "flags": [],
        },
    ]

    context = ContextBuilder().build(state)

    assert "2026-07-07T20:00:00+08:00" in context
    assert "old" not in context

