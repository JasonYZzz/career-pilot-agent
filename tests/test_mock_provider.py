import json

import pytest

from career_agent.model.mock_provider import MockLLM


@pytest.mark.asyncio
async def test_default_role_returns_legacy_text() -> None:
    result = await MockLLM().complete("hello")
    assert result.text == "I found context for that."


@pytest.mark.asyncio
async def test_planner_role_returns_step1_todo() -> None:
    context = json.dumps({"step": 1, "task": "生成职业规划"}, ensure_ascii=False)
    result = await MockLLM().complete(context, role="planner")
    payload = json.loads(result.text)
    assert payload["decision"] == "update_todo"
    assert payload["todo_update"]


@pytest.mark.asyncio
async def test_planner_role_step12_writes_report() -> None:
    context = json.dumps({"step": 12}, ensure_ascii=False)
    payload = json.loads((await MockLLM().complete(context, role="planner")).text)
    assert payload["decision"] == "call_tool"
    assert payload["tool_name"] == "write_file"
    assert payload["tool_args"]["path"] == "outputs/career_plan.md"


@pytest.mark.asyncio
async def test_planner_role_garbage_context_defaults_to_step1() -> None:
    payload = json.loads((await MockLLM().complete("not json", role="planner")).text)
    assert payload["decision"] == "update_todo"


@pytest.mark.asyncio
async def test_compression_role_returns_eleven_keys() -> None:
    payload = json.loads((await MockLLM().complete("ctx", role="compression")).text)
    expected = {"task_goal", "user_constraints", "student_profile_facts",
                "career_direction_candidates", "important_evidence",
                "loaded_skills_summary", "tool_results_summary", "todo_state",
                "open_questions", "risk_flags", "next_steps"}
    assert expected.issubset(payload.keys())


@pytest.mark.asyncio
async def test_critic_role_returns_issues_json() -> None:
    payload = json.loads((await MockLLM().complete("report", role="critic")).text)
    assert payload["severity"] in {"ok", "minor", "major"}
    assert isinstance(payload["issues"], list)


@pytest.mark.asyncio
async def test_report_role_echoes_student_profile() -> None:
    prompt = "<student_profile>\n计算机专业大三，目标 AI 应用开发。\n</student_profile>"
    report = (await MockLLM().complete(prompt, role="report")).text
    assert "计算机专业大三" in report
    assert "方向评分表" in report
    assert "能力差距矩阵" in report
    assert "每周执行清单" in report
    assert "不能保证就业、录用或薪资结果" in report
