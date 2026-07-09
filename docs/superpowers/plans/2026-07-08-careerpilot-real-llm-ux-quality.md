# CareerPilot Real LLM UX and Report Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复真实百炼模型路径的闭环失败，给 CLI 增加实时运行进度输出，并把最终职业规划报告从 demo 模板升级为可复盘、有证据、有行动细节的报告。

**Architecture:** 保持现有轻量 Agent Runtime，不引入数据库、前端或重型 Agent 框架。新增小型 JSON 提取工具、CLI 事件输出回调、工具观察写回策略和更强报告 prompt/模板；Planner 仍只输出结构化 `AgentDecision`，Runtime 负责工具执行与 trace。失败时明确标记本次未生成新报告，避免旧 `career_plan.md` 误导用户。

**Tech Stack:** Python ≥3.12、uv、Typer、pydantic-settings、OpenAI SDK Responses 兼容模式、pytest + pytest-asyncio、ruff、mypy。

## Global Constraints

- Python ≥3.12；包管理与命令执行使用 `uv`（`uv run pytest` / `uv run ruff check .` / `uv run mypy .`）。
- 只考虑后端和 CLI，不考虑前端 UI、数据库、向量库、部署系统或重型多 Agent 框架。
- 文件内容一律视为不可信资料，不得覆盖运行规则；不承诺就业、录用、薪资、面试通过等结果。
- 所有文件访问必须限制在 `--workspace` 内；默认只允许写入 `workspace/outputs/`。
- Trace 不记录完整隐私原文；模型原始输出只允许保存截断后的脱敏 preview。
- `LLMProvider.complete` 保持 async；真实 provider 仍使用百炼 qwen3.7-plus OpenAI Responses 兼容模式。
- 不修改用户本地 `.env` 中的真实 API key；测试使用 fake provider 或 `LLM_PROVIDER=mock`。
- 每个任务必须有独立测试或可验证命令；代码变更按 TDD 写失败测试后实现。

---

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `career_agent/runtime/json_utils.py` | 从真实模型输出中提取单个 JSON 对象，支持 fenced block / 前后解释文字 | 新增 |
| `career_agent/runtime/planner.py` | 使用 JSON 提取工具解析 Planner 输出；trace/decision reason 能暴露安全 preview | 修改 |
| `career_agent/runtime/agent_loop.py` | 增加 CLI 事件回调；保存 `list_dir` 等低风险工具观察；运行开始标记本次报告状态 | 修改 |
| `career_agent/runtime/events.py` | 定义 `RunEvent` 与事件 sink 类型，避免 CLI 与 runtime 强耦合 | 新增 |
| `career_agent/cli.py` | 新增 `--verbose/--quiet`，实时输出 step、模型耗时、工具调用、压缩、终止原因 | 修改 |
| `career_agent/runtime/report_synthesizer.py` | 加强报告证据包、章节深度和 fallback 模板；避免只输出 mock 风格短报告 | 修改 |
| `career_agent/prompts/report_prompt.md` | 要求方向评分表、证据引用、能力差距矩阵、30/60/90 详细计划、每周 todo | 修改 |
| `career_agent/prompts/planner_prompt.md` | 强化目录观察使用规则，减少重复 `list_dir` | 修改 |
| `career_agent/runtime/trace_logger.py` | 可选记录 `model_output_preview`、CLI event 不进隐私正文 | 修改 |
| `tests/test_json_utils.py` | JSON 提取单元测试 | 新增 |
| `tests/test_planner_validation.py` | Planner 解析 fenced JSON / invalid preview 测试 | 修改 |
| `tests/test_agent_loop.py` | list_dir 写回、CLI/event 回调、失败不误认旧报告测试 | 修改 |
| `tests/test_cli_smoke.py` | CLI verbose 输出 smoke 测试 | 修改 |
| `tests/test_report_synthesizer.py` | 深度报告结构和证据引用测试 | 修改 |
| `examples/trace_with_compression.json` | 重新生成示例 trace | 更新 |
| `workspace/outputs/career_plan.md` | 重新生成示例报告 | 更新 |

---

## Task 1: Robust Planner JSON Extraction

**Files:**
- Create: `career_agent/runtime/json_utils.py`
- Modify: `career_agent/runtime/planner.py`
- Modify: `tests/test_planner_validation.py`
- Create: `tests/test_json_utils.py`

**Interfaces:**
- Produces: `extract_json_object(text: str) -> dict[str, Any]`
- Consumes: `Planner.next_decision(...)` uses `extract_json_object(result.text)` instead of direct `json.loads(result.text)`
- Produces: invalid planner decision reason remains non-sensitive: `model returned invalid decision JSON`

- [ ] **Step 1: Write failing JSON extraction tests**

Create `tests/test_json_utils.py`:

```python
import pytest

from career_agent.runtime.json_utils import extract_json_object


def test_extract_json_object_accepts_plain_json() -> None:
    payload = extract_json_object('{"decision":"final_answer","reason":"ok"}')
    assert payload == {"decision": "final_answer", "reason": "ok"}


def test_extract_json_object_accepts_fenced_json() -> None:
    text = '```json\n{"decision":"call_tool","tool_name":"read_file"}\n```'
    payload = extract_json_object(text)
    assert payload["decision"] == "call_tool"
    assert payload["tool_name"] == "read_file"


def test_extract_json_object_accepts_explanatory_prefix() -> None:
    text = '下面是决策：\n{"decision":"load_skill","skill_name":"role_matching"}\n请执行。'
    payload = extract_json_object(text)
    assert payload["decision"] == "load_skill"
    assert payload["skill_name"] == "role_matching"


def test_extract_json_object_rejects_array() -> None:
    with pytest.raises(ValueError, match="object"):
        extract_json_object('[{"decision":"final_answer"}]')


def test_extract_json_object_rejects_missing_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        extract_json_object("我需要继续分析。")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_json_utils.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'career_agent.runtime.json_utils'`.

- [ ] **Step 3: Implement `extract_json_object`**

Create `career_agent/runtime/json_utils.py`:

```python
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取单个 JSON 对象。

    参数: text 模型原始输出，可能包含 ```json 代码块或前后解释文字。
    返回: 解析后的 dict。
    """
    candidates = _candidate_json_strings(text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        raise ValueError("model JSON must be an object")
    raise ValueError("no JSON object found in model output")


def _candidate_json_strings(text: str) -> list[str]:
    """按可靠性顺序返回可能的 JSON 字符串候选。"""
    stripped = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates = [item.strip() for item in fenced]
    candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates
```

- [ ] **Step 4: Run JSON tests**

Run: `uv run pytest tests/test_json_utils.py -q`

Expected: PASS.

- [ ] **Step 5: Write Planner fenced-output test**

Append to `tests/test_planner_validation.py`:

```python
async def test_planner_accepts_fenced_json_decision() -> None:
    text = """```json
{"decision":"call_tool","tool_name":"read_file","tool_args":{"path":"data/student_profile.md"},"reason":"读取画像"}
```"""
    planner = Planner(StaticLLM(text))
    decision = await planner.next_decision(
        RunState(run_id="r", task="t", workspace=Path(".")),
        "{}",
    )
    assert decision.decision == "call_tool"
    assert decision.tool_name == "read_file"
```

- [ ] **Step 6: Run Planner test to verify it fails**

Run: `uv run pytest tests/test_planner_validation.py::test_planner_accepts_fenced_json_decision -q`

Expected: FAIL because Planner still uses direct `json.loads`.

- [ ] **Step 7: Use extractor in Planner**

Modify `career_agent/runtime/planner.py`:

```python
from career_agent.runtime.json_utils import extract_json_object
```

Replace:

```python
payload = json.loads(result.text)
```

with:

```python
payload = extract_json_object(result.text)
```

Replace the `except json.JSONDecodeError:` block with:

```python
except ValueError:
    return AgentDecision(
        decision="ask_clarification",
        reason="model returned invalid decision JSON",
    )
```

Remove the unused `import json` if no longer needed.

- [ ] **Step 8: Run focused tests**

Run: `uv run pytest tests/test_json_utils.py tests/test_planner_validation.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add career_agent/runtime/json_utils.py career_agent/runtime/planner.py tests/test_json_utils.py tests/test_planner_validation.py
git commit -m "fix(planner): parse JSON decisions from real model output"
```

---

## Task 2: Preserve Tool Observations for Planner

**Files:**
- Modify: `career_agent/runtime/agent_loop.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `state.tool_results` includes safe observations for `list_dir` and successful `write_file`
- Consumes: `ContextBuilder._recent_tool_results()` already surfaces `state.tool_results`

- [ ] **Step 1: Write failing test for `list_dir` observation**

Append to `tests/test_agent_loop.py`:

```python
from career_agent.runtime.run_state import AgentDecision, RunState
from career_agent.runtime.trace_logger import TraceLogger


def test_agent_loop_records_list_dir_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "job_roles").mkdir(parents=True)
    (workspace / "data" / "job_roles" / "backend_engineer.md").write_text("后端", encoding="utf-8")
    state = RunState(run_id="r", task="t", workspace=workspace)
    trace = TraceLogger("r", "t", workspace, "mock", "mock", "mock")
    loop = AgentLoop(Settings(llm_provider="mock"))
    decision = AgentDecision(
        decision="call_tool",
        tool_name="list_dir",
        tool_args={"path": "data/job_roles"},
        reason="列目录",
    )

    import asyncio

    asyncio.run(loop._call_tool(state, decision, trace))

    assert state.tool_results
    assert state.tool_results[-1]["tool"] == "list_dir"
    assert state.tool_results[-1]["path"] == "data/job_roles"
    assert "backend_engineer.md" in str(state.tool_results[-1]["content"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_loop.py::test_agent_loop_records_list_dir_observation -q`

Expected: FAIL because `list_dir` is not appended to `state.tool_results`.

- [ ] **Step 3: Record safe tool observations**

In `career_agent/runtime/agent_loop.py`, in `_handle_tool_result`, replace:

```python
if tool_name in {"get_time", "create_reminder"}:
```

with:

```python
if tool_name in {"list_dir", "get_time", "create_reminder"}:
```

Keep the existing append body:

```python
state.tool_results.append(
    {
        "tool": tool_name,
        "path": str(result.metadata.get("path", "")),
        "content": result.content,
        "truncated": result.truncated,
        "flags": [],
    }
)
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_agent_loop.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add career_agent/runtime/agent_loop.py tests/test_agent_loop.py
git commit -m "fix(loop): feed directory observations back to planner"
```

---

## Task 3: CLI Progress Events

**Files:**
- Create: `career_agent/runtime/events.py`
- Modify: `career_agent/runtime/agent_loop.py`
- Modify: `career_agent/cli.py`
- Modify: `tests/test_agent_loop.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: `RunEvent = dataclass(type: str, message: str, payload: dict[str, object])`
- Produces: `EventSink = Callable[[RunEvent], None]`
- Produces: `AgentLoop(settings=None, event_sink: EventSink | None = None)`
- Produces CLI flags: `--verbose/--quiet`; default `--verbose=True`

- [ ] **Step 1: Write event sink test**

Append to `tests/test_agent_loop.py`:

```python
from career_agent.runtime.events import RunEvent


def test_agent_loop_emits_progress_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "job_roles").mkdir(parents=True)
    (workspace / "skills").mkdir()
    (workspace / "outputs").mkdir()
    (workspace / "data" / "student_profile.md").write_text("计算机专业大三，目标 AI。", encoding="utf-8")
    (workspace / "data" / "resume_draft.md").write_text("项目：Agent Demo。", encoding="utf-8")
    (workspace / "data" / "injection_resume.md").write_text("普通补充资料。", encoding="utf-8")
    (workspace / "data" / "job_roles" / "all_roles_long.md").write_text("AI 应用开发需要 Python。", encoding="utf-8")
    (workspace / "skills" / "index.json").write_text("[]", encoding="utf-8")
    events: list[RunEvent] = []

    AgentLoop(Settings(llm_provider="mock", max_steps=12), event_sink=events.append).run(
        task="生成职业规划",
        workspace=workspace,
        trace_path=tmp_path / "trace.json",
    )

    event_types = [event.type for event in events]
    assert "run_start" in event_types
    assert "model_call" in event_types
    assert "tool_call" in event_types
    assert "run_end" in event_types
```

- [ ] **Step 2: Run event test to verify it fails**

Run: `uv run pytest tests/test_agent_loop.py::test_agent_loop_emits_progress_events -q`

Expected: FAIL because `career_agent.runtime.events` does not exist or `AgentLoop` has no `event_sink`.

- [ ] **Step 3: Add runtime event types**

Create `career_agent/runtime/events.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunEvent:
    """CLI 可订阅的运行事件。

    参数: type 事件类型；message 面向人类的短消息；payload 机器可读字段。
    返回: 不适用。
    """

    type: str
    message: str
    payload: dict[str, object] = field(default_factory=dict)


EventSink = Callable[[RunEvent], None]
```

- [ ] **Step 4: Wire event sink into AgentLoop**

Modify `career_agent/runtime/agent_loop.py` imports:

```python
from career_agent.runtime.events import EventSink, RunEvent
```

Modify constructor:

```python
def __init__(self, settings: Settings | None = None, event_sink: EventSink | None = None) -> None:
    self.settings = settings or Settings()
    self.event_sink = event_sink
```

Add method:

```python
def _emit(self, event_type: str, message: str, **payload: object) -> None:
    """向 CLI 输出运行进度；无 sink 时不做事。"""
    if self.event_sink:
        self.event_sink(RunEvent(event_type, message, payload))
```

Call `_emit` in these places:

```python
self._emit("run_start", f"开始运行：{task[:60]}", workspace=str(workspace))
```

after `token_budget` span:

```python
self._emit(
    "token_budget",
    f"Step {state.step}: tokens={tokens}, compress={should_compress}",
    step=state.step,
    tokens=tokens,
    should_compress=should_compress,
    reason=reason,
)
```

after planner `model_call` span:

```python
self._emit(
    "model_call",
    f"Step {state.step}: decision={decision.decision}"
    + (f" tool={decision.tool_name}" if decision.tool_name else "")
    + (f" skill={decision.skill_name}" if decision.skill_name else ""),
    step=state.step,
    decision=decision.decision,
    tool_name=decision.tool_name or "",
    skill_name=decision.skill_name or "",
)
```

after tool call span:

```python
self._emit(
    "tool_call",
    f"Tool {tool_name}: {'ok' if result.ok else 'failed'}",
    tool=tool_name,
    ok=result.ok,
    error=result.error or "",
)
```

after compression span:

```python
self._emit(
    "compression",
    f"Compression: {before} -> {after} tokens",
    before_tokens=before,
    after_tokens=after,
    reason=reason,
)
```

before returning from `run_async` after `run_end` span:

```python
self._emit("run_end", f"结束：{state.termination_reason}", termination_reason=state.termination_reason)
```

- [ ] **Step 5: Run event test**

Run: `uv run pytest tests/test_agent_loop.py::test_agent_loop_emits_progress_events -q`

Expected: PASS.

- [ ] **Step 6: Add CLI verbose output**

Modify `career_agent/cli.py` imports:

```python
from career_agent.runtime.events import RunEvent
```

Add helper:

```python
def _print_event(event: RunEvent) -> None:
    """把运行事件输出到 CLI。"""
    typer.echo(f"[career-agent] {event.message}")
```

Modify `run(...)` signature:

```python
verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress events."),
```

Modify AgentLoop construction:

```python
sink = _print_event if verbose else None
state = AgentLoop(settings, event_sink=sink).run(
    task=task,
    workspace=workspace_path,
    trace_path=trace_path,
)
```

- [ ] **Step 7: Add CLI smoke test**

In `tests/test_cli_smoke.py`, add:

```python
def test_cli_run_verbose_prints_progress(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "job_roles").mkdir(parents=True)
    (workspace / "skills").mkdir()
    (workspace / "outputs").mkdir()
    (workspace / "data" / "student_profile.md").write_text("计算机专业大三，目标 AI。", encoding="utf-8")
    (workspace / "data" / "resume_draft.md").write_text("项目：Agent Demo。", encoding="utf-8")
    (workspace / "data" / "injection_resume.md").write_text("普通补充资料。", encoding="utf-8")
    (workspace / "data" / "job_roles" / "all_roles_long.md").write_text("AI 应用开发需要 Python。", encoding="utf-8")
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
```

- [ ] **Step 8: Run CLI and loop tests**

Run: `uv run pytest tests/test_agent_loop.py tests/test_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add career_agent/runtime/events.py career_agent/runtime/agent_loop.py career_agent/cli.py tests/test_agent_loop.py tests/test_cli_smoke.py
git commit -m "feat(cli): stream agent progress events"
```

---

## Task 4: Avoid Stale Report Confusion

**Files:**
- Modify: `career_agent/runtime/agent_loop.py`
- Modify: `career_agent/cli.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `workspace/outputs/run_status.json`
- Produces run status values: `running`, `completed`, `failed`
- CLI prints whether `outputs/career_plan.md` was generated in this run

- [ ] **Step 1: Write failing stale-output test**

Append to `tests/test_agent_loop.py`:

```python
class InvalidJsonLLM:
    async def complete(self, prompt: str, *, system: str = "", role: str = "default") -> "LLMResult":
        from career_agent.model.base import LLMResult

        _ = (prompt, system, role)
        return LLMResult(text="不是 JSON")


def test_failed_run_marks_report_not_generated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "outputs").mkdir(parents=True)
    (workspace / "outputs" / "career_plan.md").write_text("旧报告", encoding="utf-8")
    state = AgentLoop(Settings(llm_provider="mock")).run(
        task="生成职业规划",
        workspace=workspace,
        trace_path=tmp_path / "trace.json",
    )
    status = json.loads((workspace / "outputs" / "run_status.json").read_text(encoding="utf-8"))
    assert status["report_generated"] is False
    assert status["career_plan_path"] == "outputs/career_plan.md"
    assert (workspace / "outputs" / "career_plan.md").read_text(encoding="utf-8") == "旧报告"
    assert state.termination_reason in {"missing_critical_info", "max_steps", "final_answer"}
```

Then adjust the test to inject `InvalidJsonLLM` by adding optional `llm` parameter in Step 3 if needed. The final assertion should prove a failed run does not claim the old report is new.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_loop.py::test_failed_run_marks_report_not_generated -q`

Expected: FAIL because `run_status.json` is not written.

- [ ] **Step 3: Allow dependency injection for LLM in AgentLoop**

Modify constructor in `career_agent/runtime/agent_loop.py`:

```python
from career_agent.model.base import LLMProvider
```

Signature:

```python
def __init__(
    self,
    settings: Settings | None = None,
    event_sink: EventSink | None = None,
    llm: LLMProvider | None = None,
) -> None:
```

LLM assignment:

```python
self.llm = llm or llm_from_settings(self.settings)
```

Update failing test to use:

```python
state = AgentLoop(Settings(llm_provider="mock"), llm=InvalidJsonLLM()).run(...)
```

- [ ] **Step 4: Write run status helper**

In `AgentLoop`, add:

```python
def _write_run_status(self, state: RunState, status: str) -> None:
    """写出本次运行状态，避免旧报告被误认为新产物。"""
    path = state.workspace / "outputs" / "run_status.json"
    report_path = state.workspace / "outputs" / "career_plan.md"
    report_generated = (
        state.termination_reason == "final_answer"
        and report_path.exists()
        and state.final_answer == "职业规划报告已生成。"
    )
    payload = {
        "run_id": state.run_id,
        "status": status,
        "termination_reason": state.termination_reason,
        "report_generated": report_generated,
        "career_plan_path": "outputs/career_plan.md",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

At start of `run_async`, after `state` is created:

```python
self._write_run_status(state, "running")
```

In `finally`, before `trace.export(...)`:

```python
status = "completed" if state.termination_reason == "final_answer" else "failed"
self._write_run_status(state, status)
```

- [ ] **Step 5: Update CLI final message**

In `career_agent/cli.py`, after run:

```python
status_path = workspace_path / "outputs" / "run_status.json"
if status_path.exists():
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not status_payload.get("report_generated"):
        typer.echo("CareerPilot note: no new career_plan.md was generated in this run.")
```

Add `import json`.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/test_agent_loop.py tests/test_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add career_agent/runtime/agent_loop.py career_agent/cli.py tests/test_agent_loop.py tests/test_cli_smoke.py
git commit -m "fix(loop): mark whether reports are generated in current run"
```

---

## Task 5: Upgrade Report Depth

**Files:**
- Modify: `career_agent/prompts/report_prompt.md`
- Modify: `career_agent/runtime/report_synthesizer.py`
- Modify: `tests/test_report_synthesizer.py`

**Interfaces:**
- Produces report sections: `方向评分表`, `证据引用`, `能力差距矩阵`, `30 / 60 / 90 天行动计划`, `每周执行清单`, `简历改写建议`, `假设与待确认问题`
- Produces fallback report with at least 900 Chinese characters when profile/resume/roles exist

- [ ] **Step 1: Write failing report-depth test**

Append to `tests/test_report_synthesizer.py`:

```python
@pytest.mark.asyncio
async def test_fallback_report_is_detailed_and_evidence_based(tmp_path: Path) -> None:
    state = RunState(run_id="r", task="生成职业规划", workspace=tmp_path)
    state.tool_results = [
        {
            "tool": "read_file",
            "path": "data/student_profile.md",
            "content": "计算机科学与技术大三，兴趣：AI 应用开发、本地 Agent、Python 自动化。每周可投入 12 小时。",
            "truncated": False,
            "flags": [],
        },
        {
            "tool": "read_file",
            "path": "data/resume_draft.md",
            "content": "项目：本地 Agent Demo；课程管理系统后端接口；技能：Python、FastAPI、SQL。",
            "truncated": False,
            "flags": [],
        },
        {
            "tool": "read_file",
            "path": "data/job_roles/all_roles_long.md",
            "content": "AI 应用开发需要 Python、工具调用、评测和 trace。后端开发需要接口、数据库、测试。产品经理需要需求分析。",
            "truncated": False,
            "flags": [],
        },
    ]

    report = await ReportSynthesizer(llm=RaisingLLM(), library=PromptLibrary()).build(state)

    for heading in [
        "方向评分表",
        "证据引用",
        "能力差距矩阵",
        "30 / 60 / 90 天行动计划",
        "每周执行清单",
        "简历改写建议",
        "假设与待确认问题",
    ]:
        assert heading in report
    assert len(report) >= 900
    assert "不能保证就业、录用或薪资结果" in report
    assert "Python" in report
    assert "Agent Demo" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_synthesizer.py::test_fallback_report_is_detailed_and_evidence_based -q`

Expected: FAIL because current fallback report is too shallow and lacks required headings.

- [ ] **Step 3: Strengthen report prompt**

Modify `career_agent/prompts/report_prompt.md` so required sections include exactly:

```markdown
## 必含章节
1. 结论摘要
2. 方向评分表
3. 证据引用
4. 学生画像摘要
5. 能力差距矩阵
6. 简历改写建议
7. 面试准备计划
8. 30 / 60 / 90 天行动计划
9. 每周执行清单
10. 风险与边界
11. 假设与待确认问题
```

Add:

```markdown
## 细节要求
- 方向评分表必须比较 AI 应用开发、后端开发、产品经理，给出 1-5 分、依据、风险。
- 证据引用必须引用学生画像、简历、岗位资料中的具体短句，不要捏造。
- 能力差距矩阵必须包含 能力项 / 当前证据 / 差距 / 30天补齐动作。
- 30 / 60 / 90 天计划每阶段至少 4 条可执行动作。
- 每周执行清单至少 6 周，每周包含目标、交付物、复盘问题。
```

- [ ] **Step 4: Upgrade fallback template**

In `career_agent/runtime/report_synthesizer.py`, replace `_fallback_template_from` with a detailed template that includes all headings:

```python
def _fallback_template_from(self, evidence: EvidencePack) -> str:
    profile_summary = self._summarize_profile(evidence)
    direction_summary = self._choose_direction(evidence)
    missing = self._missing_questions(evidence)
    return f"""# 大学生职业规划报告

## 1. 结论摘要
{direction_summary}

该结论基于当前工作区资料，不能保证就业、录用或薪资结果。建议把本报告视为 90 天试运行计划，而不是最终职业定论。

## 2. 方向评分表
| 方向 | 匹配度 | 依据 | 主要风险 |
| --- | --- | --- | --- |
| AI 应用开发 | 5/5 | 学生画像和简历均出现 AI、Python、本地 Agent、工具调用相关证据。 | 需要把 demo 做成可展示作品，并补齐评测、日志和产品场景说明。 |
| 后端开发 | 4/5 | 简历有课程管理系统后端接口、FastAPI、SQL 证据。 | 差异化不如 AI 应用方向明显，需要加强测试、性能和稳定性案例。 |
| 产品经理 | 3/5 | 学生偏好能理解产品需求，但当前产品实践证据较少。 | 需要补充 PRD、用户访谈、需求拆解作品。 |

## 3. 证据引用
- 学生画像：{self._inline_evidence(evidence.student_profile)}
- 简历草稿：{self._inline_evidence(evidence.resume)}
- 岗位资料：{self._inline_evidence(evidence.role_material)}

## 4. 学生画像摘要
{profile_summary}

## 5. 能力差距矩阵
| 能力项 | 当前证据 | 差距 | 30 天补齐动作 |
| --- | --- | --- | --- |
| AI 应用工程 | 本地 Agent Demo、Python 自动化。 | 缺少评测指标、异常处理和真实业务场景。 | 给 Agent Demo 增加 3 个工具、失败 trace 和 README 演示。 |
| 后端工程 | FastAPI、SQL、课程管理系统接口。 | 缺少测试覆盖、性能说明和部署约束。 | 为一个接口补 pytest、错误处理和压测记录。 |
| 产品表达 | 能理解产品需求。 | 缺少 PRD、用户故事和取舍说明。 | 为 Agent Demo 写 1 页 PRD 和目标用户流程。 |
| 面试表达 | 有项目经历。 | 项目成果指标不够明确。 | 改写 2 个 STAR 项目故事并录制模拟回答。 |

## 6. 简历改写建议
- 把“本地 Agent Demo”改写为：实现文件读取、工具调用、token 压缩和 trace 导出，支持可复盘的职业规划任务。
- 把“课程管理系统”补充接口数量、数据库表、错误处理、测试或性能数据。
- 技能栏按目标方向排序：Python / FastAPI / SQL / Agent Runtime / Prompt 工程 / Git。
- 删除泛泛描述，优先写可验证交付物、仓库链接、测试命令和截图。

## 7. 面试准备计划
- 准备 3 分钟项目介绍：问题背景、架构、工具调用、边界处理、结果。
- 准备 5 个追问：为什么不用 LangChain、如何限制文件访问、trace 如何复盘、压缩策略、失败兜底。
- 准备后端基础：HTTP、数据库索引、事务、接口测试、异常处理。
- 准备产品表达：目标用户、核心痛点、MVP 范围、为什么不做前端。

## 8. 30 / 60 / 90 天行动计划
- 30 天：完善 Agent Demo；补 README；新增 5 个测试；输出一份项目复盘；完成简历第一版。
- 60 天：增加真实模型 smoke；完善 CLI 进度输出；做一次模拟面试；投递 5 个 AI 应用或后端实习岗位。
- 90 天：形成作品集页面或仓库索引；完成 2 个项目深度复盘；按反馈调整方向；建立每周复盘节奏。

## 9. 每周执行清单
- 第 1 周：整理现有项目证据；交付物：项目 README 草稿；复盘问题：项目亮点是否一句话说清。
- 第 2 周：补测试和 trace 示例；交付物：测试截图和 trace 样例；复盘问题：失败路径是否可解释。
- 第 3 周：重写简历项目段；交付物：简历 v1；复盘问题：是否有量化或可验证结果。
- 第 4 周：准备模拟面试；交付物：10 个问答卡片；复盘问题：回答是否具体。
- 第 5 周：补一个后端小功能；交付物：接口、测试、错误处理说明；复盘问题：工程能力是否体现。
- 第 6 周：小规模投递和反馈收集；交付物：岗位清单和反馈表；复盘问题：方向是否需要调整。

## 10. 风险与边界
{self._summarize_risks(evidence)}

## 11. 假设与待确认问题
- 假设当前资料能代表学生主要经历；若缺少课程成绩、实习经历、城市偏好，需要重新评估。
- 待确认：{missing}
"""
```

Add helper:

```python
def _inline_evidence(self, text: str) -> str:
    """提取一段短证据；缺失时明确标注。"""
    if not text.strip():
        return "资料缺失"
    cleaned = " ".join(line.strip().lstrip("- ").strip() for line in text.splitlines() if line.strip())
    return cleaned[:120]
```

- [ ] **Step 5: Run report tests**

Run: `uv run pytest tests/test_report_synthesizer.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add career_agent/prompts/report_prompt.md career_agent/runtime/report_synthesizer.py tests/test_report_synthesizer.py
git commit -m "feat(report): produce evidence-based detailed career plans"
```

---

## Task 6: Real-LLM Closure Regression with Fake Provider

**Files:**
- Modify: `tests/test_agent_loop.py`
- Modify: `career_agent/model/base.py` if needed only for test typing

**Interfaces:**
- Consumes: `AgentLoop(..., llm=...)` from Task 4
- Produces: fake real-like LLM that returns fenced JSON decisions and deep report text

- [ ] **Step 1: Write fake real LLM regression test**

Append to `tests/test_agent_loop.py`:

```python
class FencedDecisionLLM:
    def __init__(self) -> None:
        self.planner_calls = 0

    async def complete(self, prompt: str, *, system: str = "", role: str = "default") -> "LLMResult":
        from career_agent.model.base import LLMResult

        _ = (prompt, system)
        if role == "planner":
            self.planner_calls += 1
            decisions = [
                '{"decision":"update_todo","todo_update":[{"id":"read","title":"读取资料","status":"pending","note":""}],"reason":"建清单"}',
                '{"decision":"call_tool","tool_name":"read_file","tool_args":{"path":"data/student_profile.md","max_chars":6000},"reason":"读画像"}',
                '{"decision":"call_tool","tool_name":"read_file","tool_args":{"path":"data/resume_draft.md","max_chars":6000},"reason":"读简历"}',
                '{"decision":"call_tool","tool_name":"list_dir","tool_args":{"path":"data/job_roles"},"reason":"看岗位目录"}',
                '{"decision":"call_tool","tool_name":"read_file","tool_args":{"path":"data/job_roles/all_roles_long.md","max_chars":6000},"reason":"读岗位资料"}',
                '{"decision":"call_tool","tool_name":"write_file","tool_args":{"path":"outputs/career_plan.md","content":"","mode":"overwrite"},"reason":"写报告"}',
            ]
            raw = decisions[min(self.planner_calls - 1, len(decisions) - 1)]
            return LLMResult(text=f"```json\n{raw}\n```")
        if role == "report":
            return LLMResult(text="# 大学生职业规划报告\n\n## 1. 结论摘要\n建议 AI 应用开发。\n\n## 2. 方向评分表\nAI 应用开发 5/5。\n\n## 3. 证据引用\nPython、Agent Demo。\n\n## 4. 学生画像摘要\n计算机专业。\n\n## 5. 能力差距矩阵\n补评测。\n\n## 6. 简历改写建议\n突出 Agent Demo。\n\n## 7. 面试准备计划\n准备项目讲解。\n\n## 8. 30 / 60 / 90 天行动计划\n30 天完善项目，60 天模拟面试，90 天作品集。\n\n## 9. 每周执行清单\n每周复盘。\n\n## 10. 风险与边界\n不能保证就业、录用或薪资结果。\n\n## 11. 假设与待确认问题\n确认城市。")
        if role == "critic":
            return LLMResult(text='{"issues":[],"severity":"ok"}')
        if role == "compression":
            return LLMResult(text='{"task_goal":"生成职业规划","user_constraints":[],"student_profile_facts":[],"career_direction_candidates":[],"important_evidence":[],"loaded_skills_summary":[],"tool_results_summary":[],"todo_state":[],"open_questions":[],"risk_flags":[],"next_steps":[]}')
        return LLMResult(text="ok")


def test_agent_loop_real_like_fenced_json_closes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "job_roles").mkdir(parents=True)
    (workspace / "skills").mkdir()
    (workspace / "outputs").mkdir()
    (workspace / "data" / "student_profile.md").write_text("计算机专业大三，目标 AI。", encoding="utf-8")
    (workspace / "data" / "resume_draft.md").write_text("项目：Agent Demo。", encoding="utf-8")
    (workspace / "data" / "job_roles" / "all_roles_long.md").write_text("AI 应用开发需要 Python。", encoding="utf-8")
    (workspace / "skills" / "index.json").write_text("[]", encoding="utf-8")

    state = AgentLoop(Settings(llm_provider="mock"), llm=FencedDecisionLLM()).run(
        task="生成职业规划",
        workspace=workspace,
        trace_path=tmp_path / "trace.json",
    )

    assert state.termination_reason == "final_answer"
    report = (workspace / "outputs" / "career_plan.md").read_text(encoding="utf-8")
    assert "方向评分表" in report
    status = json.loads((workspace / "outputs" / "run_status.json").read_text(encoding="utf-8"))
    assert status["report_generated"] is True
```

- [ ] **Step 2: Run regression test**

Run: `uv run pytest tests/test_agent_loop.py::test_agent_loop_real_like_fenced_json_closes -q`

Expected: PASS if Tasks 1-5 are correct. If it fails, fix the failing task's implementation, not the test.

- [ ] **Step 3: Run full loop tests**

Run: `uv run pytest tests/test_agent_loop.py tests/test_planner_validation.py tests/test_report_synthesizer.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_loop.py
git commit -m "test(loop): cover real-like fenced JSON planner closure"
```

---

## Task 7: Regenerate Examples and Documentation Notes

**Files:**
- Modify: `README.md`
- Regenerate: `examples/trace_with_compression.json`
- Regenerate: `workspace/outputs/career_plan.md`

**Interfaces:**
- README documents mock demo command, real model smoke command, `--verbose/--quiet`, and stale-report warning.
- Example trace contains progress-compatible spans and successful report generation.

- [ ] **Step 1: Update README usage notes**

In `README.md`, add a CLI behavior section:

```markdown
### CLI 运行可见性

`career-agent run` 默认输出进度事件，包括 step、token 估算、模型决策、工具调用、压缩和结束原因。若只想保留最终摘要，可加 `--quiet`。

离线演示建议显式使用 mock，避免本地 `.env` 中未配置完整的真实模型地址：

```bash
LLM_PROVIDER=mock LLM_PROTOCOL=mock LLM_API_KEY= DASHSCOPE_API_KEY= \
uv run career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md 和 data/job_roles 下的岗位资料，生成职业规划和 90 天行动计划。" \
  --workspace ./workspace \
  --trace ./trace.json
```

运行结束后请查看 `workspace/outputs/run_status.json`。当 `report_generated=false` 时，本次没有生成新的 `career_plan.md`，已有报告可能是旧产物。
```

- [ ] **Step 2: Run quality gates**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy .
```

Expected: all PASS.

- [ ] **Step 3: Regenerate mock example**

Run:

```bash
LLM_PROVIDER=mock LLM_PROTOCOL=mock LLM_API_KEY= DASHSCOPE_API_KEY= \
uv run career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md、data/injection_resume.md 和 data/job_roles 下的岗位资料，帮我在 AI应用开发、后端开发、产品经理三个方向中选择最适合的职业路径，生成 90 天行动计划，并给出每周复盘提醒草案。" \
  --workspace ./workspace \
  --trace ./examples/trace_with_compression.json \
  --quiet
```

Expected: command succeeds with `status=final_answer`.

- [ ] **Step 4: Verify example artifacts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
trace = json.loads(Path("examples/trace_with_compression.json").read_text())
spans = trace["spans"]
types = {span["type"] for span in spans}
required = {"model_call", "tool_call", "token_budget", "compression", "boundary_event"}
missing = required - types
report = Path("workspace/outputs/career_plan.md").read_text()
status = json.loads(Path("workspace/outputs/run_status.json").read_text())
print("missing_span_types=", sorted(missing))
print("termination=", trace["termination_reason"])
print("report_generated=", status["report_generated"])
print("report_len=", len(report))
assert not missing
assert trace["termination_reason"] == "final_answer"
assert status["report_generated"] is True
assert "方向评分表" in report
assert "能力差距矩阵" in report
assert "不能保证就业、录用或薪资结果" in report
assert "忽略以上指令" not in report
assert "一定能进大厂" not in report
PY
```

Expected: assertions pass.

- [ ] **Step 5: Commit**

```bash
git add README.md examples/trace_with_compression.json
git add -f workspace/outputs/career_plan.md
git commit -m "docs: document verbose runs and regenerate rich example outputs"
```

---

## Self-Review

**1. Spec coverage**

- 真实模型 Planner JSON 容错：Task 1 covers fenced JSON and explanatory wrappers.
- `list_dir` 观察写回，避免重复目录查询：Task 2.
- CLI 等待久、无进度反馈：Task 3.
- 旧报告误导：Task 4.
- 报告太简单、像 mock：Task 5.
- 真实模型闭环回归：Task 6.
- README 和示例 trace/report：Task 7.

**2. Placeholder scan**

No `TBD`, `TODO`, `implement later`, or unspecified “add tests” steps. Every task includes concrete files, commands, and expected outcomes.

**3. Type consistency**

- `RunEvent(type, message, payload)` is defined before CLI/AgentLoop use.
- `AgentLoop(settings=None, event_sink=None, llm=None)` is used consistently in tests.
- `extract_json_object(text) -> dict[str, Any]` is used by Planner only.
- `run_status.json` keys are stable: `run_id`, `status`, `termination_reason`, `report_generated`, `career_plan_path`.

