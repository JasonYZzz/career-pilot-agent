import json
from pathlib import Path

from career_agent.runtime.boundary_guard import BoundaryGuard
from career_agent.runtime.run_state import RunState
from career_agent.tools.registry import build_default_tool_registry


def test_file_tools_list_read_write(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / "data").mkdir()
    (workspace / "outputs").mkdir()
    (workspace / "data" / "student_profile.md").write_text("计算机专业大三", encoding="utf-8")
    state = RunState(run_id="run_test", task="职业规划", workspace=workspace)
    registry = build_default_tool_registry(BoundaryGuard())

    listed = registry.run("list_dir", {"path": "data"}, state)
    assert listed.ok
    assert "student_profile.md" in listed.content

    read = registry.run("read_file", {"path": "data/student_profile.md", "max_chars": 20}, state)
    assert read.ok
    assert "计算机专业" in read.content

    written = registry.run(
        "write_file",
        {"path": "outputs/career_plan.md", "content": "# 报告", "mode": "overwrite"},
        state,
    )
    assert written.ok
    assert (workspace / "outputs" / "career_plan.md").exists()


def test_todo_and_reminder_tools(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    state = RunState(run_id="run_test", task="每周提醒", workspace=tmp_path)
    registry = build_default_tool_registry(BoundaryGuard())

    todo = registry.run(
        "todo_update",
        {"items": [{"id": "read", "title": "读取资料", "status": "done", "note": ""}]},
        state,
    )
    assert todo.ok
    assert state.todos[0]["status"] == "done"

    reminder = registry.run(
        "create_reminder",
        {"title": "每周复盘", "date": "2026-07-14", "note": "复盘投递", "confirmed": False},
        state,
    )
    assert reminder.ok
    data = json.loads((tmp_path / "outputs" / "reminder_plan.json").read_text(encoding="utf-8"))
    assert data[0]["status"] == "draft_requires_confirmation"


def test_reminder_ignores_model_confirmed_true(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    state = RunState(run_id="run_test", task="每周提醒", workspace=tmp_path)
    registry = build_default_tool_registry(BoundaryGuard())

    result = registry.run(
        "create_reminder",
        {"title": "每周复盘", "date": "2026-07-14", "note": "复盘投递", "confirmed": True},
        state,
    )

    assert result.ok
    data = json.loads((tmp_path / "outputs" / "reminder_plan.json").read_text(encoding="utf-8"))
    assert data[0]["status"] == "draft_requires_confirmation"
