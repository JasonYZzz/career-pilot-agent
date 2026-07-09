from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from career_agent.model.base import LLMProvider
    from career_agent.prompts.library import PromptLibrary


class Critic:
    """报告质量审查：关键词预检 + LLM 复审，失败回落仅关键词。"""

    required_sections = (
        "结论摘要",
        "学生画像摘要",
        "方向比较",
        "能力差距",
        "30 / 60 / 90 天行动计划",
        "需要用户进一步确认的信息",
    )
    overclaim_phrases = ("一定能", "保证就业", "保证录用", "保证薪资", "必进大厂")

    def __init__(
        self,
        llm: "LLMProvider | None" = None,
        library: "PromptLibrary | None" = None,
        llm_timeout_seconds: float = 30.0,
    ) -> None:
        """
        初始化报告质量审查器。

        Args:
            llm: 可选的 LLM 提供者，用于深度审查
            library: 可选的提示词库，提供系统提示词

        Returns:
            None
        """
        self.llm = llm
        self.library = library
        self.llm_timeout_seconds = llm_timeout_seconds

    async def check_report(self, markdown: str) -> list[str]:
        """
        检查报告质量问题。

        先执行关键词预检，若有 LLM 则追加 LLM 审查结果。
        LLM 调用失败时自动回落到仅关键词结果。

        Args:
            markdown: 待检查的 Markdown 报告文本

        Returns:
            问题列表，每个问题为字符串描述
        """
        issues = self._keyword_issues(markdown)
        if self.llm and self.library:
            try:
                issues.extend(await self._llm_issues(markdown))
            except Exception:
                pass  # 回落仅关键词
        return issues

    def _keyword_issues(self, markdown: str) -> list[str]:
        """
        快速关键词预检：缺章节与过度承诺。

        Args:
            markdown: 待检查的 Markdown 报告文本

        Returns:
            问题列表，格式为 "type:detail"
        """
        issues = []
        for section in self.required_sections:
            if section not in markdown:
                issues.append(f"missing_section:{section}")
        for phrase in self.overclaim_phrases:
            if phrase in markdown and f"不{phrase}" not in markdown and f"不能{phrase}" not in markdown:
                issues.append(f"overclaim:{phrase}")
        return issues

    async def _llm_issues(self, markdown: str) -> list[str]:
        """
        调用 LLM 取问题清单，转为字符串列表。

        Args:
            markdown: 待检查的 Markdown 报告文本

        Returns:
            问题列表，从 LLM 返回的 JSON 中提取
        """
        assert self.llm is not None
        assert self.library is not None
        result = await asyncio.wait_for(
            self.llm.complete(markdown, system=self.library.system_for("critic"), role="critic"),
            timeout=self.llm_timeout_seconds,
        )
        payload = json.loads(result.text)
        return [str(item) for item in payload.get("issues", [])]
