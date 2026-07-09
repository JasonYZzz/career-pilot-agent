from __future__ import annotations

from pathlib import Path

from career_agent.runtime.token_budget import estimate_tokens
from career_agent.skills.registry import SkillRegistry


class SkillLoader:
    def load(self, name: str, registry: SkillRegistry, workspace: Path) -> str:
        meta = registry.get(name)
        if meta is None:
            raise ValueError(f"unknown skill: {name}")
        skill_path = (workspace / meta.path).resolve()
        workspace_resolved = workspace.resolve()
        if workspace_resolved not in skill_path.parents and skill_path != workspace_resolved:
            raise ValueError(f"skill path outside workspace: {meta.path}")
        content = skill_path.read_text(encoding="utf-8")
        if estimate_tokens(content) <= meta.max_tokens:
            return content
        max_chars = max(1000, meta.max_tokens * 4)
        return content[:max_chars] + "\n\n[skill truncated by token cap]"

