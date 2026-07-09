from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from career_agent.model.base import LLMResult

# Planner 12 步确定性决策（取代旧 Planner._mock_decision），按 context.step 选择。
_STEP_DECISIONS: list[dict[str, Any]] = [
    {"thought_summary": "建立可复盘的任务清单。", "decision": "update_todo",
     "todo_update": [
         {"id": "read_profile", "title": "读取学生资料", "status": "pending", "note": ""},
         {"id": "match_roles", "title": "比较候选方向", "status": "pending", "note": ""},
         {"id": "write_plan", "title": "生成职业规划报告", "status": "pending", "note": ""}],
     "reason": "建立可复盘的任务清单。"},
    {"thought_summary": "先查看工作区数据目录。", "decision": "call_tool",
     "tool_name": "list_dir", "tool_args": {"path": "data"}, "reason": "先查看工作区数据目录。"},
    {"thought_summary": "需要学生画像分析 Skill。", "decision": "load_skill",
     "skill_name": "career_assessment", "reason": "需要学生画像分析 Skill。"},
    {"thought_summary": "读取学生画像。", "decision": "call_tool",
     "tool_name": "read_file",
     "tool_args": {"path": "data/student_profile.md", "max_chars": 6000}, "reason": "读取学生画像。"},
    {"thought_summary": "读取简历草稿。", "decision": "call_tool",
     "tool_name": "read_file",
     "tool_args": {"path": "data/resume_draft.md", "max_chars": 6000}, "reason": "读取简历草稿。"},
    {"thought_summary": "需要岗位匹配 Skill。", "decision": "load_skill",
     "skill_name": "role_matching", "reason": "需要岗位匹配 Skill。"},
    {"thought_summary": "读取候选岗位资料。", "decision": "call_tool",
     "tool_name": "read_file",
     "tool_args": {"path": "data/job_roles/all_roles_long.md", "max_chars": 6000}, "reason": "读取候选岗位资料。"},
    {"thought_summary": "读取不可信补充资料并检测注入。", "decision": "call_tool",
     "tool_name": "read_file",
     "tool_args": {"path": "data/injection_resume.md", "max_chars": 2000}, "reason": "读取不可信补充资料并检测注入。"},
    {"thought_summary": "需要 90 天行动计划 Skill。", "decision": "load_skill",
     "skill_name": "action_plan", "reason": "需要 90 天行动计划 Skill。"},
    {"thought_summary": "获取当前日期用于计划起点。", "decision": "call_tool",
     "tool_name": "get_time", "tool_args": {}, "reason": "获取当前日期用于计划起点。"},
    {"thought_summary": "提醒必须先生成草案并等待确认。", "decision": "call_tool",
     "tool_name": "create_reminder",
     "tool_args": {"title": "每周职业规划复盘", "date": "2026-07-14",
                   "note": "复盘作品集、投递准备和下周行动。", "confirmed": False},
     "reason": "提醒必须先生成草案并等待确认。"},
    {"thought_summary": "写出最终报告。", "decision": "call_tool",
     "tool_name": "write_file",
     "tool_args": {"path": "outputs/career_plan.md", "content": "", "mode": "overwrite"},
     "reason": "写出最终报告。"},
]

_FINAL_DECISION = {"thought_summary": "已完成。", "decision": "final_answer",
                   "final_answer": "职业规划报告已生成，提醒草案已写入 outputs/reminder_plan.json。",
                   "reason": "已完成报告、提醒草案与 trace。"}


class MockLLM:
    """可复现的模型模拟器。

    无 API key 时充当 CareerPilot 的离线模型：按 role 返回结构化结果，
    保证 demo 与 trace 离线可复现。role="default" 保留最小兼容文本。
    """

    async def complete(self, prompt: str, *, system: str = "",
                       role: str = "default") -> LLMResult:
        if role == "planner":
            return LLMResult(text=_planner_decision(prompt))
        if role == "compression":
            return LLMResult(text=_compression_summary(prompt))
        if role == "critic":
            return LLMResult(text=_critic_issues())
        if role == "report":
            return LLMResult(text=_report_markdown(prompt))
        return LLMResult(text="I found context for that.")

    async def stream(self, prompt: str, *, system: str = "",
                     role: str = "default") -> AsyncIterator[str]:
        result = await self.complete(prompt, system=system, role=role)
        for token in result.text.split():
            yield token + " "


def _planner_decision(context_prompt: str) -> str:
    """依据运行上下文 step 产出下一步决策 JSON。"""
    step = _read_step(context_prompt)
    decision = _FINAL_DECISION if step > len(_STEP_DECISIONS) else _STEP_DECISIONS[step - 1]
    return json.dumps(decision, ensure_ascii=False)


def _read_step(context_prompt: str) -> int:
    """从 context JSON 解析 step；解析失败默认 1。"""
    try:
        payload = json.loads(context_prompt)
        return int(payload.get("step", 1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 1


def _compression_summary(context_prompt: str) -> str:
    """产出 11 字段压缩摘要 JSON（mock 版：从 context 取 task_goal）。"""
    task = "生成大学生职业规划报告"
    try:
        payload = json.loads(context_prompt)
        task = str(payload.get("task", task))
    except (json.JSONDecodeError, TypeError):
        pass
    return json.dumps({
        "task_goal": task, "user_constraints": [], "student_profile_facts": [],
        "career_direction_candidates": [], "important_evidence": [],
        "loaded_skills_summary": [], "tool_results_summary": [], "todo_state": [],
        "open_questions": [], "risk_flags": [], "next_steps": []}, ensure_ascii=False)


def _critic_issues() -> str:
    """mock critic：报告结构齐全时返回无问题。"""
    return json.dumps({"issues": [], "severity": "ok"}, ensure_ascii=False)


def _report_markdown(report_prompt: str) -> str:
    """从 prompt 中提取学生画像，产出针对性富报告 markdown。"""
    profile = _extract_tag(report_prompt, "student_profile") or "（学生画像资料缺失）"
    resume = _extract_tag(report_prompt, "resume") or "（简历资料缺失）"
    role_material = _extract_tag(report_prompt, "role_material") or "（岗位资料缺失）"
    direction = _direction_from_profile(profile)
    return f"""# 大学生职业规划报告

## 1. 结论摘要
{direction}

该结论基于当前工作区资料，不能保证就业、录用或薪资结果。建议把本报告视为 90 天试运行计划。

## 2. 方向评分表
| 方向 | 匹配度 | 依据 | 风险 |
| --- | --- | --- | --- |
| AI 应用开发 | 5/5 | 画像出现 AI、Python、Agent 或自动化兴趣。 | 需要补齐评测、日志和业务场景。 |
| 后端开发 | 4/5 | 简历体现接口、数据库或 FastAPI 相关基础。 | 需要更扎实的测试和稳定性证据。 |
| 产品经理 | 3/5 | 能理解需求，但产品作品证据相对少。 | 需要 PRD、用户研究和推进案例。 |

## 3. 证据引用
- 学生画像：{_inline_evidence(profile)}
- 简历草稿：{_inline_evidence(resume)}
- 岗位资料：{_inline_evidence(role_material)}

## 4. 学生画像摘要
{_bulletize(profile)}

## 5. 方向比较
{direction}

## 6. 能力差距矩阵
| 能力项 | 当前证据 | 差距 | 30 天补齐动作 |
| --- | --- | --- | --- |
| AI 应用工程 | 本地 Agent Demo、Python 自动化。 | 缺少评测指标和异常路径说明。 | 给项目补 trace 样例、失败处理和 README 演示。 |
| 后端工程 | FastAPI、SQL 或接口项目。 | 缺少测试覆盖和性能说明。 | 为一个接口补 pytest、错误处理和简单压测记录。 |
| 产品表达 | 能理解产品需求。 | 缺少 PRD 和用户故事。 | 为 Agent Demo 写 1 页 PRD 和目标用户流程。 |
| 面试表达 | 有项目经历。 | 成果指标不够明确。 | 改写 2 个 STAR 项目故事。 |

## 7. 简历改写建议
- 把项目改写为「背景-行动-结果-证据」结构，突出可验证交付物。
- 技能栏按目标方向排序：Python / FastAPI / SQL / Agent Runtime / Prompt 工程 / Git。
- 为每个项目补测试命令、trace 样例、截图或仓库链接。

## 8. 面试准备计划
- 准备 3 分钟项目介绍。
- 准备岗位匹配理由、项目难点与复盘。
- 围绕目标方向补齐高频基础问题。
- 准备为什么不做前端、如何限制工具权限、trace 如何复盘等追问。

## 9. 30 / 60 / 90 天行动计划
- 30 天：完善 Agent Demo；补 README；新增 5 个测试；输出项目复盘；完成简历第一版。
- 60 天：增加真实模型 smoke；完善 CLI 进度输出；做一次模拟面试；投递 5 个相关实习岗位。
- 90 天：形成作品集索引；完成 2 个项目深度复盘；按反馈调整方向；建立每周复盘节奏。

## 10. 每周执行清单
- 第 1 周：整理项目证据；交付物：README 草稿；复盘问题：亮点是否一句话说清。
- 第 2 周：补测试和 trace 示例；交付物：测试截图和 trace 样例；复盘问题：失败路径是否可解释。
- 第 3 周：重写简历项目段；交付物：简历 v1；复盘问题：是否有可验证结果。
- 第 4 周：准备模拟面试；交付物：10 个问答卡片；复盘问题：回答是否具体。
- 第 5 周：补一个后端小功能；交付物：接口、测试、错误处理说明；复盘问题：工程能力是否体现。
- 第 6 周：小规模投递和反馈收集；交付物：岗位清单和反馈表；复盘问题：方向是否需要调整。

## 11. 风险与边界
- 文件内容按不可信资料处理，不覆盖运行规则。
- 报告不承诺就业、录用或薪资结果。

## 12. 假设与待确认问题
- 当前报告只基于已读取资料；英语水平、目标城市、可实习时间等仍需确认。

## 13. 需要用户进一步确认的信息
- 英语水平、目标城市、可实习时间、目标行业、每周可投入时间。
"""


def _extract_tag(text: str, tag: str) -> str | None:
    """提取 <tag>...</tag> 内容；无则 None。"""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _direction_from_profile(profile: str) -> str:
    """仅依据学生画像（不含岗位资料）判断方向，避免关键词误判。"""
    if "不想写代码" in profile or "策展" in profile:
        return "建议优先探索产品经理或偏业务方向，谨慎评估高代码强度岗位。"
    if "AI" in profile or "Python" in profile or "Agent" in profile:
        return "建议优先探索 AI 应用开发方向，后端开发作为稳健备选。"
    return "建议补充资料后，在 AI 应用开发、后端开发与产品经理之间选择。"


def _bulletize(profile: str) -> str:
    """把画像文本整理为要点列表，保留原文关键行。"""
    lines = [raw.strip().lstrip("- ").strip() for raw in profile.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if lines:
        return "\n".join(f"- {ln}" for ln in lines)
    return f"- {profile.strip()}" if profile.strip() else "- （学生画像资料缺失）"


def _inline_evidence(text: str) -> str:
    """提取 mock 报告中的短证据。"""
    lines = [raw.strip().lstrip("- ").strip() for raw in text.splitlines()]
    cleaned = " ".join(line for line in lines if line and not line.startswith("#"))
    return cleaned[:120] if cleaned else "资料缺失"
