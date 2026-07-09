# career_agent/prompts/library.py
from __future__ import annotations

from pathlib import Path


class PromptLibrary:
    """加载并缓存 career_agent/prompts 下的 markdown prompt。

    用途：集中管理各角色 system prompt，供 Planner / Compressor /
    ReportSynthesizer / Critic 复用，避免 prompt 散落在代码字符串里。

    参数: prompts_dir 指定 prompt 目录，默认为本文件同级目录。
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or Path(__file__).resolve().parent
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        """返回角色 prompt 原文；name 对应文件 {name}_prompt.md。"""
        if name not in self._cache:
            path = self._dir / f"{name}_prompt.md"
            if not path.is_file():
                raise KeyError(f"prompt not found: {name} ({path})")
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]

    def system_for(self, role: str) -> str:
        """组装「共享运行规则 + 角色指令」作为 LLM 的 system 消息。"""
        return f"{self.get('system')}\n\n---\n\n{self.get(role)}"
