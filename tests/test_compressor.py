from pathlib import Path

import pytest

from career_agent.model.base import LLMResult
from career_agent.model.mock_provider import MockLLM
from career_agent.prompts.library import PromptLibrary
from career_agent.runtime.compressor import ContextCompressor
from career_agent.runtime.run_state import RunState


@pytest.mark.asyncio
async def test_compressor_preserves_required_keys(tmp_path: Path) -> None:
    state = RunState(run_id="run_test", task="生成职业规划报告", workspace=tmp_path)
    state.todos = [{"id": "read_profile", "status": "done"}]
    state.loaded_skills = {"career_assessment": "分析学生画像"}
    state.tool_results = [
        {"tool": "read_file", "path": "data/student_profile.md", "content": "计算机专业大三"}
    ]
    summary = await ContextCompressor().compress(state)

    assert summary["task_goal"] == "生成职业规划报告"
    assert "todo_state" in summary
    assert "loaded_skills_summary" in summary
    assert "tool_results_summary" in summary


class RaisingLLM:
    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        role: str = "default",
    ) -> LLMResult:
        _ = (prompt, system, role)
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_compressor_uses_llm_when_available(tmp_path: Path) -> None:
    state = RunState(run_id="r", task="生成职业规划报告", workspace=tmp_path)
    summary = await ContextCompressor(MockLLM(), PromptLibrary()).compress(state)
    assert summary["task_goal"] == "生成职业规划报告"
    assert "next_steps" in summary  # 来自 mock 11 字段


@pytest.mark.asyncio
async def test_compressor_falls_back_on_llm_failure(tmp_path: Path) -> None:
    state = RunState(run_id="r", task="生成职业规划报告", workspace=tmp_path)
    state.todos = [{"id": "x", "status": "pending"}]
    summary = await ContextCompressor(RaisingLLM(), PromptLibrary()).compress(state)
    assert summary["task_goal"] == "生成职业规划报告"  # 规则兜底仍可用
    assert summary["todo_state"] == state.todos
