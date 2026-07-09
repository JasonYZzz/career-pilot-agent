from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillMeta:
    name: str
    path: str
    description: str
    triggers: list[str]
    max_tokens: int


@dataclass(frozen=True)
class SkillRegistry:
    skills: list[SkillMeta]

    @classmethod
    def load_index(cls, workspace: Path) -> "SkillRegistry":
        index_path = workspace / "skills" / "index.json"
        if not index_path.exists():
            return cls(skills=[])
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return cls(
            skills=[
                SkillMeta(
                    name=str(item["name"]),
                    path=str(item["path"]),
                    description=str(item.get("description", "")),
                    triggers=[str(trigger) for trigger in item.get("triggers", [])],
                    max_tokens=int(item.get("max_tokens", 1200)),
                )
                for item in data
            ]
        )

    def get(self, name: str) -> SkillMeta | None:
        return next((skill for skill in self.skills if skill.name == name), None)

