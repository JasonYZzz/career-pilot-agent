import asyncio
import json
from pathlib import Path

import typer

from career_agent.config import Settings, resolve_workspace
from career_agent.model.factory import llm_from_settings
from career_agent.runtime.agent_loop import AgentLoop
from career_agent.runtime.events import RunEvent

app = typer.Typer(help="CareerPilot Agent command line interface.")


def _print_event(event: RunEvent) -> None:
    """把运行事件输出到 CLI。"""
    typer.echo(f"[career-agent] {event.message}")


@app.command()
def run(
    task: str = typer.Option(..., "--task", help="User task for the career planning agent."),
    workspace: str = typer.Option("./workspace", "--workspace", help="Workspace directory."),
    trace: str = typer.Option("./trace.json", "--trace", help="Trace output path."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress events."),
) -> None:
    settings = Settings()
    workspace_path = resolve_workspace(workspace)
    trace_path = Path(trace).expanduser().resolve()
    sink = _print_event if verbose else None
    state = AgentLoop(settings, event_sink=sink).run(
        task=task,
        workspace=workspace_path,
        trace_path=trace_path,
    )
    status_path = workspace_path / "outputs" / "run_status.json"
    if status_path.exists():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        if not status_payload.get("report_generated", True):
            typer.echo("CareerPilot note: no new career_plan.md was generated in this run.")
    typer.echo(f"CareerPilot completed: status={state.termination_reason} trace={trace_path}")


@app.command("llm-smoke")
def llm_smoke(
    prompt: str = typer.Option("用一句话回复 OK", "--prompt"),
    system: str = typer.Option("You are a concise assistant.", "--system"),
) -> None:
    settings = Settings()
    llm = llm_from_settings(settings)
    result = asyncio.run(llm.complete(prompt, system=system))
    typer.echo(result.text)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    typer.echo(f"Local service mode is planned for {host}:{port}; CLI run is the MVP path.")
