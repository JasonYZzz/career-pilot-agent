# tests/test_planner.py
import json
from pathlib import Path

import pytest

from career_agent.model.mock_provider import MockLLM
from career_agent.prompts.library import PromptLibrary
from career_agent.runtime.planner import Planner
from career_agent.runtime.run_state import RunState


@pytest.mark.asyncio
async def test_mock_planner_returns_structured_decision(tmp_path: Path) -> None:
    state = RunState(run_id="run_test", task="请生成职业规划报告", workspace=tmp_path)
    state.step = 1
    context = json.dumps({"step": 1, "task": state.task}, ensure_ascii=False)
    decision = await Planner(MockLLM(), PromptLibrary()).next_decision(state, context)
    assert decision.decision == "update_todo"
    assert decision.reason
