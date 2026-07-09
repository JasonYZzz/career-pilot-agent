from pathlib import Path

from typer.testing import CliRunner

from career_agent.cli import app

runner = CliRunner()


def test_cli_help_renders_run_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "CareerPilot" in result.stdout


def test_cli_run_verbose_prints_progress(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "job_roles").mkdir(parents=True)
    (workspace / "skills").mkdir()
    (workspace / "outputs").mkdir()
    (workspace / "data" / "student_profile.md").write_text(
        "计算机专业大三，目标 AI。",
        encoding="utf-8",
    )
    (workspace / "data" / "resume_draft.md").write_text("项目：Agent Demo。", encoding="utf-8")
    (workspace / "data" / "injection_resume.md").write_text("普通补充资料。", encoding="utf-8")
    (workspace / "data" / "job_roles" / "all_roles_long.md").write_text(
        "AI 应用开发需要 Python。",
        encoding="utf-8",
    )
    (workspace / "skills" / "index.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "生成职业规划",
            "--workspace",
            str(workspace),
            "--trace",
            str(tmp_path / "trace.json"),
            "--verbose",
        ],
        env={"LLM_PROVIDER": "mock", "LLM_PROTOCOL": "mock", "LLM_API_KEY": "", "DASHSCOPE_API_KEY": ""},
    )

    assert result.exit_code == 0
    assert "[career-agent] Step" in result.stdout
    assert "Tool" in result.stdout
