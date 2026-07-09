from __future__ import annotations

from career_agent.skills.registry import SkillMeta


def select_skills(
    task: str,
    metas: list[SkillMeta],
    already_loaded: set[str],
    limit: int,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for meta in metas:
        if meta.name in already_loaded:
            continue
        score = sum(2 for trigger in meta.triggers if trigger and trigger in task)
        if meta.description and any(token in task for token in meta.description.split("、")):
            score += 1
        if score > 0:
            scored.append((score, meta.name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]

