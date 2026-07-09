import json
from pathlib import Path

from career_agent.skills.loader import SkillLoader
from career_agent.skills.registry import SkillRegistry
from career_agent.skills.selector import select_skills


def test_skill_index_and_loader(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "name": "career_assessment",
                    "path": "skills/career_assessment.md",
                    "description": "分析学生画像",
                    "triggers": ["职业规划", "方向选择"],
                    "max_tokens": 1200,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (skills_dir / "career_assessment.md").write_text("# Career Assessment\n分析规则", encoding="utf-8")

    registry = SkillRegistry.load_index(tmp_path)
    selected = select_skills("请做职业规划和方向选择", registry.skills, set(), limit=1)
    assert selected == ["career_assessment"]
    content = SkillLoader().load("career_assessment", registry, tmp_path)
    assert "分析规则" in content
