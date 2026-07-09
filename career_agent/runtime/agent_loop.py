from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from career_agent.config import Settings
from career_agent.model.base import LLMProvider
from career_agent.model.factory import llm_from_settings
from career_agent.prompts.library import PromptLibrary
from career_agent.runtime.boundary_guard import BoundaryGuard
from career_agent.runtime.compressor import ContextCompressor
from career_agent.runtime.context_builder import ContextBuilder
from career_agent.runtime.critic import Critic
from career_agent.runtime.events import EventSink, RunEvent
from career_agent.runtime.planner import Planner
from career_agent.runtime.report_synthesizer import ReportSynthesizer
from career_agent.runtime.run_state import AgentDecision, RunState, ToolResult
from career_agent.runtime.token_budget import TokenBudgetManager, estimate_tokens
from career_agent.runtime.trace_logger import TraceLogger
from career_agent.skills.loader import SkillLoader
from career_agent.skills.registry import SkillRegistry
from career_agent.tools.registry import build_default_tool_registry


class AgentLoop:
    def __init__(
        self,
        settings: Settings | None = None,
        event_sink: EventSink | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.event_sink = event_sink
        self.guard = BoundaryGuard()
        self.budget = TokenBudgetManager(self.settings.token_budget())
        # library 与 llm 必须先于消费方创建，供 Compressor/Critic/ReportSynthesizer/Planner 复用。
        self.library = PromptLibrary()
        self.llm = llm or llm_from_settings(self.settings)
        self.compressor = ContextCompressor(self.llm, self.library)
        self.context_builder = ContextBuilder()
        self.critic = Critic(self.llm, self.library)
        self.report_synthesizer = ReportSynthesizer(self.critic, self.llm, self.library)
        self.planner = Planner(self.llm, self.library)
        self.skill_loader = SkillLoader()
        self.tools = build_default_tool_registry(self.guard)

    def _emit(self, event_type: str, message: str, **payload: object) -> None:
        """向 CLI 输出运行进度；无 sink 时不做事。"""
        if self.event_sink:
            self.event_sink(RunEvent(event_type, message, payload))

    def run(self, task: str, workspace: Path, trace_path: Path) -> RunState:
        return asyncio.run(self.run_async(task=task, workspace=workspace, trace_path=trace_path))

    async def run_async(self, task: str, workspace: Path, trace_path: Path) -> RunState:
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "outputs").mkdir(parents=True, exist_ok=True)
        state = RunState(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            task=task,
            workspace=workspace,
            max_steps=self.settings.max_steps,
        )
        self._write_run_status(state, "running")
        skills = SkillRegistry.load_index(workspace)
        trace = TraceLogger(
            run_id=state.run_id,
            task=task,
            workspace=workspace,
            llm_provider=self.settings.llm_provider,
            llm_protocol=self.settings.llm_protocol,
            llm_model=self.settings.llm_model,
        )
        trace.add_span("run_start", workspace=str(workspace), task_preview=task[:160])
        self._emit("run_start", f"开始运行：{task[:60]}", workspace=str(workspace))
        try:
            while not state.done and state.step < state.max_steps:
                state.step += 1
                context = self.context_builder.build(state)
                tokens = estimate_tokens(context)
                should_compress, reason = self.budget.should_compress(
                    context,
                    loaded_skill_count=len(state.loaded_skills),
                    step=state.step,
                )
                trace.add_span(
                    "token_budget",
                    step=state.step,
                    estimated_input_tokens=tokens,
                    max_context_tokens=self.settings.max_context_tokens,
                    should_compress=should_compress,
                    reason=reason,
                )
                self._emit(
                    "token_budget",
                    f"Step {state.step}: tokens={tokens}, compress={should_compress}",
                    step=state.step,
                    tokens=tokens,
                    should_compress=should_compress,
                    reason=reason,
                )
                if should_compress and state.compressed_summary is None:
                    await self._compress(state, trace, reason)
                    context = self.context_builder.build(state)

                started = time.perf_counter()
                decision = await self.planner.next_decision(state, context)
                trace.add_span(
                    "model_call",
                    name="planner",
                    step=state.step,
                    decision=decision.decision,
                    tool_name=decision.tool_name,
                    skill_name=decision.skill_name,
                    reason=decision.reason,
                    estimated_input_tokens=estimate_tokens(context),
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
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
                await self._apply_decision(state, decision, skills, trace)

            if not state.done:
                state.done = True
                state.termination_reason = "max_steps"
                state.final_answer = "达到最大步骤数，已导出当前 trace。"
            trace.add_span("run_end", termination_reason=state.termination_reason)
            self._emit(
                "run_end",
                f"结束：{state.termination_reason}",
                termination_reason=state.termination_reason or "",
            )
            return state
        finally:
            status = "completed" if state.termination_reason == "final_answer" else "failed"
            self._write_run_status(state, status)
            trace.export(trace_path, state)

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

    async def _apply_decision(
        self,
        state: RunState,
        decision: AgentDecision,
        skills: SkillRegistry,
        trace: TraceLogger,
    ) -> None:
        if decision.decision == "update_todo":
            state.todos = decision.todo_update or []
            trace.add_span("todo_update", count=len(state.todos))
            return
        if decision.decision == "load_skill":
            self._load_skill(state, decision.skill_name or "", skills, trace)
            return
        if decision.decision == "compress_context":
            await self._compress(state, trace, decision.reason)
            return
        if decision.decision == "call_tool":
            await self._call_tool(state, decision, trace)
            return
        if decision.decision == "final_answer":
            state.done = True
            state.final_answer = decision.final_answer or ""
            state.termination_reason = "final_answer"
            trace.add_span("final_answer", preview=state.final_answer[:200])
            return
        if decision.decision == "ask_clarification":
            state.done = True
            state.final_answer = decision.reason
            state.termination_reason = "missing_critical_info"
            trace.add_span("final_answer", preview=decision.reason[:200])

    def _load_skill(
        self,
        state: RunState,
        name: str,
        skills: SkillRegistry,
        trace: TraceLogger,
    ) -> None:
        started = time.perf_counter()
        try:
            content = self.skill_loader.load(name, skills, state.workspace)
            state.loaded_skills[name] = content
            trace.add_span(
                "skill_load",
                name=name,
                estimated_tokens=estimate_tokens(content),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            self._boundary_event(state, trace, "skill_load_failed", str(exc))

    async def _call_tool(self, state: RunState, decision: AgentDecision, trace: TraceLogger) -> None:
        tool_name = decision.tool_name or ""
        args = dict(decision.tool_args or {})
        if tool_name == "write_file" and args.get("path") == "outputs/career_plan.md":
            args["content"] = await self._build_report(state)
        result = self.tools.run(tool_name, args, state)
        trace.add_span(
            "tool_call",
            name=tool_name,
            ok=result.ok,
            error=result.error,
            truncated=result.truncated,
            elapsed_ms=result.elapsed_ms,
            metadata=result.metadata,
        )
        self._emit(
            "tool_call",
            f"Tool {tool_name}: {'ok' if result.ok else 'failed'}",
            tool=tool_name,
            ok=result.ok,
            error=result.error or "",
        )
        self._handle_tool_result(state, trace, tool_name, args, result)

    def _handle_tool_result(
        self,
        state: RunState,
        trace: TraceLogger,
        tool_name: str,
        args: dict[str, object],
        result: ToolResult,
    ) -> None:
        if result.ok:
            state.repeated_failures.pop(json.dumps([tool_name, args], sort_keys=True), None)
        else:
            key = json.dumps([tool_name, args], sort_keys=True)
            state.repeated_failures[key] = state.repeated_failures.get(key, 0) + 1
            self._boundary_event(state, trace, "tool_failure", result.error or "unknown tool error")
            if state.repeated_failures[key] > 2:
                state.done = True
                state.termination_reason = "repeated_tool_failure"
            return

        if tool_name in {"list_dir", "get_time", "create_reminder"}:
            state.tool_results.append(
                {
                    "tool": tool_name,
                    "path": str(result.metadata.get("path", "")),
                    "content": result.content,
                    "truncated": result.truncated,
                    "flags": [],
                }
            )
        if tool_name == "read_file":
            flags = result.metadata.get("flags", [])
            for flag in flags:
                self._boundary_event(state, trace, flag, str(result.metadata.get("path", "")))
        if tool_name == "create_reminder":
            self._boundary_event(
                state,
                trace,
                "reminder_requires_confirmation",
                "draft written, no real reminder created",
            )
        if tool_name == "write_file" and args.get("path") == "outputs/career_plan.md":
            state.done = True
            state.final_answer = "职业规划报告已生成。"
            state.termination_reason = "final_answer"

    async def _compress(self, state: RunState, trace: TraceLogger, reason: str) -> None:
        started = time.perf_counter()
        # 压缩前后各测算一次 token，便于通过 trace 观察 11 字段摘要对上下文规模的影响。
        before = estimate_tokens(self.context_builder.build(state))
        state.compressed_summary = await self.compressor.compress(state)
        state.last_compression_tool_result_count = len(state.tool_results)
        after = estimate_tokens(self.context_builder.build(state))
        trace.add_span(
            "compression",
            reason=reason,
            before_tokens=before,
            after_tokens=after,
            watermark=self.settings.compression_watermark,
            summary_keys=list(state.compressed_summary),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        self._emit(
            "compression",
            f"Compression: {before} -> {after} tokens",
            before_tokens=before,
            after_tokens=after,
            reason=reason,
        )

    def _boundary_event(
        self,
        state: RunState,
        trace: TraceLogger,
        event_type: str,
        detail: str,
    ) -> None:
        event = {"type": event_type, "detail": detail[:300]}
        state.boundary_events.append(event)
        trace.add_span("boundary_event", event_type=event_type, detail=event["detail"])

    async def _build_report(self, state: RunState) -> str:
        return await self.report_synthesizer.build(state)
