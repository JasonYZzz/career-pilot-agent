# tests/test_prompt_library.py
import pytest

from career_agent.prompts.library import PromptLibrary


def test_prompt_library_loads_all_roles() -> None:
    lib = PromptLibrary()
    for name in ["system", "planner", "compression", "critic", "report"]:
        text = lib.get(name)
        assert text.strip(), f"prompt {name} 为空"


def test_prompt_library_system_for_combines_preamble_and_role() -> None:
    combined = PromptLibrary().system_for("planner")
    assert "CareerPilot" in combined
    assert "Planner" in combined or "规划" in combined


def test_prompt_library_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        PromptLibrary().get("nonexistent")


def test_planner_prompt_tells_model_to_write_report_after_evidence() -> None:
    planner_prompt = PromptLibrary().get("planner")
    assert "已包含学生画像、简历草稿和岗位资料" in planner_prompt
    assert "outputs/career_plan.md" in planner_prompt
