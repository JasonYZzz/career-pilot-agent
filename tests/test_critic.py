import asyncio

import pytest

from career_agent.model.base import LLMResult
from career_agent.model.mock_provider import MockLLM
from career_agent.prompts.library import PromptLibrary
from career_agent.runtime.critic import Critic


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


class SlowLLM:
    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        role: str = "default",
    ) -> LLMResult:
        _ = (prompt, system, role)
        await asyncio.sleep(1)
        return LLMResult(text='{"issues":["too late"],"severity":"major"}')


@pytest.mark.asyncio
async def test_critic_keyword_flags_missing_section() -> None:
    issues = await Critic().check_report("# 报告\n只有结论，缺章节")
    assert any("missing_section" in i for i in issues)


@pytest.mark.asyncio
async def test_critic_llm_path_returns_list() -> None:
    issues = await Critic(MockLLM(), PromptLibrary()).check_report("# 完整报告 ...")
    assert isinstance(issues, list)


@pytest.mark.asyncio
async def test_critic_falls_back_on_llm_failure() -> None:
    issues = await Critic(RaisingLLM(), PromptLibrary()).check_report("# 缺章节")
    assert isinstance(issues, list)  # 关键词兜底


@pytest.mark.asyncio
async def test_critic_falls_back_on_llm_timeout() -> None:
    issues = await Critic(
        SlowLLM(),
        PromptLibrary(),
        llm_timeout_seconds=0.01,
    ).check_report("# 缺章节")
    assert isinstance(issues, list)
    assert "too late" not in issues
