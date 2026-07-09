from __future__ import annotations

import json
from typing import Any

from career_agent.model.base import LLMProvider
from career_agent.prompts.library import PromptLibrary
from career_agent.runtime.run_state import RunState

_REQUIRED_KEYS = (
    "task_goal", "user_constraints", "student_profile_facts",
    "career_direction_candidates", "important_evidence", "loaded_skills_summary",
    "tool_results_summary", "todo_state", "open_questions", "risk_flags", "next_steps",
)


class ContextCompressor:
    """上下文压缩：LLM 优先产出 11 字段摘要，失败回落规则摘要。

    用途：将运行时状态压缩为精简的结构化摘要，供 LLM 决策使用。

    参数:
        llm: LLM 提供者，可选。为 None 时仅使用规则摘要。
        library: Prompt 库，可选。为 None 时仅使用规则摘要。

    返回:
        包含 11 个必选字段的摘要字典。
    """

    def __init__(self, llm: LLMProvider | None = None,
                 library: PromptLibrary | None = None) -> None:
        """初始化压缩器。

        参数:
            llm: LLM 提供者，可选。提供时优先使用 LLM 压缩。
            library: Prompt 库，可选。提供时从库中读取 system prompt。
        """
        self.llm = llm
        self.library = library

    async def compress(self, state: RunState) -> dict[str, Any]:
        """压缩运行时状态为结构化摘要。

        优先使用 LLM 产出摘要，失败时回落到规则摘要。

        参数:
            state: 运行时状态对象。

        返回:
            包含 11 个必选字段的摘要字典。
        """
        if self.llm and self.library:
            try:
                return await self._compress_with_llm(state)
            except Exception:
                pass  # 回落规则摘要
        return self._rule_summary(state)

    async def _compress_with_llm(self, state: RunState) -> dict[str, Any]:
        """调用 LLM 产出结构化摘要；缺失字段用规则摘要补齐。

        参数:
            state: 运行时状态对象。

        返回:
            合并后的摘要字典，LLM 输出覆盖规则摘要的同名字段。
        """
        assert self.llm is not None, "llm required for LLM compression"
        assert self.library is not None, "library required for LLM compression"
        prompt = self._build_prompt(state)
        result = await self.llm.complete(
            prompt, system=self.library.system_for("compression"), role="compression")
        payload = json.loads(result.text)
        merged = self._rule_summary(state)
        for key in _REQUIRED_KEYS:
            if key in payload:
                merged[key] = payload[key]
        return merged

    def _build_prompt(self, state: RunState) -> str:
        """把当前上下文序列化为供压缩的 prompt。

        参数:
            state: 运行时状态对象。

        返回:
            JSON 格式的上下文字符串。
        """
        return json.dumps({
            "task": state.task, "loaded_skills": list(state.loaded_skills),
            "tool_results": state.tool_results, "todos": state.todos,
            "boundary_events": state.boundary_events,
        }, ensure_ascii=False)

    def _rule_summary(self, state: RunState) -> dict[str, Any]:
        return {
            "task_goal": state.task,
            "user_constraints": self._extract_constraints(state),
            "student_profile_facts": self._summarize_tool_results(
                state,
                keywords=["student_profile", "resume", "course", "project"],
            ),
            "career_direction_candidates": self._summarize_tool_results(
                state,
                keywords=["job_roles", "backend", "ai_application", "product", "AI 应用"],
            ),
            "important_evidence": self._summarize_tool_results(
                state,
                keywords=["data/", "job_roles/"],
            ),
            "loaded_skills_summary": [
                {"name": name, "summary": content[:500]}
                for name, content in state.loaded_skills.items()
            ],
            "tool_results_summary": self._summarize_tool_results(state, keywords=[]),
            "todo_state": state.todos,
            "open_questions": [],
            "risk_flags": state.boundary_events,
            "next_steps": [
                item
                for item in state.todos
                if item.get("status") in {"pending", "in_progress", "blocked"}
            ],
        }

    def _extract_constraints(self, state: RunState) -> list[str]:
        constraints = []
        for marker in ["AI应用开发", "AI 应用开发", "后端开发", "产品经理", "90 天", "提醒"]:
            if marker in state.task:
                constraints.append(marker)
        return constraints

    def _summarize_tool_results(
        self,
        state: RunState,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        rows = []
        for result in state.tool_results:
            path = str(result.get("path", ""))
            content = str(result.get("content", ""))
            if not keywords or any(keyword in path or keyword in content for keyword in keywords):
                rows.append(
                    {
                        "tool": result.get("tool"),
                        "path": path,
                        "summary": content[:700],
                        "truncated": len(content) > 700,
                    }
                )
        return rows[:12]

