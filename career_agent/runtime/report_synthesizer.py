from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from career_agent.runtime.critic import Critic
from career_agent.runtime.run_state import RunState

if TYPE_CHECKING:
    # 仅用于类型注解，运行时不导入以避免潜在循环依赖。
    from career_agent.model.base import LLMProvider
    from career_agent.prompts.library import PromptLibrary


@dataclass(frozen=True)
class EvidencePack:
    student_profile: str = ""
    resume: str = ""
    role_material: str = ""
    project_material: str = ""
    risk_flags: tuple[str, ...] = ()


class ReportSynthesizer:
    def __init__(
        self,
        critic: Critic | None = None,
        llm: "LLMProvider | None" = None,
        library: "PromptLibrary | None" = None,
        llm_timeout_seconds: float = 60.0,
    ) -> None:
        # critic 立即可用；llm/library 在 Task 7/8 接入 LLM 时使用，本任务仅存储。
        self.critic = critic or Critic()
        self.llm = llm
        self.library = library
        self.llm_timeout_seconds = llm_timeout_seconds

    async def build(self, state: RunState) -> str:
        # 先汇聚证据，再交由 LLM 起草；LLM 不可用或失败时回落到本地模板。
        # 最后用 critic 复核，若有问题追加第 13 节提示。
        evidence = self._collect_evidence(state)
        report = await self._draft_report(evidence)
        issues = await self.critic.check_report(report)
        if issues:
            report += "\n\n## 13. 质量检查提示\n" + "\n".join(f"- {issue}" for issue in issues)
        return report

    async def _draft_report(self, evidence: EvidencePack) -> str:
        """LLM 起草报告；不可用或失败时回落本地模板。

        参数: evidence 已汇聚的证据包。
        返回: 报告 markdown 文本。
        """
        if self.llm and self.library:
            try:
                result = await asyncio.wait_for(
                    self.llm.complete(
                        self._report_prompt(evidence),
                        system=self.library.system_for("report"),
                        role="report",
                    ),
                    timeout=self.llm_timeout_seconds,
                )
                if result.text.strip():
                    return result.text.strip()
            except Exception:
                # LLM 失败属于可恢复场景：静默回落模板，保证报告始终产出。
                pass
        return self._fallback_template_from(evidence)

    def _report_prompt(self, evidence: EvidencePack) -> str:
        """用标签包裹证据，供模型/mock 解析。

        参数: evidence 已汇聚的证据包。
        返回: 含 <student_profile>/<resume>/<role_material>/<project_material>
              等标签的 prompt 字符串；role_material 截断到 2000 字以防 prompt 过长。
        """
        return (
            f"<student_profile>\n{evidence.student_profile or '（缺失）'}\n</student_profile>\n"
            f"<resume>\n{evidence.resume or '（缺失）'}\n</resume>\n"
            f"<role_material>\n{evidence.role_material[:2000] or '（缺失）'}\n</role_material>\n"
            f"<project_material>\n{evidence.project_material or '（缺失）'}\n</project_material>\n"
            "<loaded_skills>\n请结合已加载 Skill 指令产出报告。\n</loaded_skills>"
        )

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

## 5. 方向比较
{self._compare_roles(evidence)}

## 6. 能力差距矩阵
| 能力项 | 当前证据 | 差距 | 30 天补齐动作 |
| --- | --- | --- | --- |
| AI 应用工程 | 本地 Agent Demo、Python 自动化。 | 缺少评测指标、异常处理和真实业务场景。 | 给 Agent Demo 增加 3 个工具、失败 trace 和 README 演示。 |
| 后端工程 | FastAPI、SQL、课程管理系统接口。 | 缺少测试覆盖、性能说明和部署约束。 | 为一个接口补 pytest、错误处理和压测记录。 |
| 产品表达 | 能理解产品需求。 | 缺少 PRD、用户故事和取舍说明。 | 为 Agent Demo 写 1 页 PRD 和目标用户流程。 |
| 面试表达 | 有项目经历。 | 项目成果指标不够明确。 | 改写 2 个 STAR 项目故事并录制模拟回答。 |

## 7. 简历改写建议
- 把“本地 Agent Demo”改写为：实现文件读取、工具调用、token 压缩和 trace 导出，支持可复盘的职业规划任务。
- 把“课程管理系统”补充接口数量、数据库表、错误处理、测试或性能数据。
- 技能栏按目标方向排序：Python / FastAPI / SQL / Agent Runtime / Prompt 工程 / Git。
- 删除泛泛描述，优先写可验证交付物、仓库链接、测试命令和截图。

## 8. 面试准备计划
- 准备一个 3 分钟项目介绍。
- 准备岗位匹配理由、项目难点、失败复盘和下一步改进。
- 围绕目标方向补齐 5 个高频基础问题。
- 准备产品表达：目标用户、核心痛点、MVP 范围、为什么不做前端。

## 9. 30 / 60 / 90 天行动计划
- 30 天：完善 Agent Demo；补 README；新增 5 个测试；输出一份项目复盘；完成简历第一版。
- 60 天：增加真实模型 smoke；完善 CLI 进度输出；做一次模拟面试；投递 5 个 AI 应用或后端实习岗位。
- 90 天：形成作品集页面或仓库索引；完成 2 个项目深度复盘；按反馈调整方向；建立每周复盘节奏。

## 10. 每周执行清单
- 第 1 周：整理现有项目证据；交付物：项目 README 草稿；复盘问题：项目亮点是否一句话说清。
- 第 2 周：补测试和 trace 示例；交付物：测试截图和 trace 样例；复盘问题：失败路径是否可解释。
- 第 3 周：重写简历项目段；交付物：简历 v1；复盘问题：是否有量化或可验证结果。
- 第 4 周：准备模拟面试；交付物：10 个问答卡片；复盘问题：回答是否具体。
- 第 5 周：补一个后端小功能；交付物：接口、测试、错误处理说明；复盘问题：工程能力是否体现。
- 第 6 周：小规模投递和反馈收集；交付物：岗位清单和反馈表；复盘问题：方向是否需要调整。

## 11. 提醒草案
如需提醒，本运行只生成提醒草案，必须经过用户确认后才能创建真实提醒。

## 12. 风险与边界
{self._summarize_risks(evidence)}

## 13. 假设与待确认问题

- 假设当前资料能代表学生主要经历；若缺少课程成绩、实习经历、城市偏好，需要重新评估。
- 待确认：{missing}

## 14. 需要用户进一步确认的信息
{missing}
"""

    def _inline_evidence(self, text: str) -> str:
        """提取一段短证据；缺失时明确标注。"""
        if not text.strip():
            return "资料缺失"
        lines = [line.strip().lstrip("- ").strip() for line in text.splitlines() if line.strip()]
        cleaned = " ".join(lines)
        return cleaned[:120]

    def _collect_evidence(self, state: RunState) -> EvidencePack:
        chunks: dict[str, list[str]] = {
            "student_profile": [],
            "resume": [],
            "role_material": [],
            "project_material": [],
        }
        risk_flags: list[str] = []
        for result in state.tool_results:
            # 先判 flags：被标记 prompt_injection_detected 的条目视为不可信，整条跳过，
            # 既不进入分块也不计入 risk_flags，避免注入文本污染报告证据。
            flags = [str(flag) for flag in result.get("flags", [])]
            if "prompt_injection_detected" in flags:
                continue
            path = str(result.get("path", ""))
            content = str(result.get("content", ""))
            risk_flags.extend(flags)
            if "student_profile" in path:
                chunks["student_profile"].append(content)
            elif "resume" in path:
                chunks["resume"].append(content)
            elif "job_roles" in path:
                chunks["role_material"].append(content)
            elif "project" in path:
                chunks["project_material"].append(content)
        return EvidencePack(
            student_profile="\n".join(chunks["student_profile"]).strip(),
            resume="\n".join(chunks["resume"]).strip(),
            role_material="\n".join(chunks["role_material"]).strip(),
            project_material="\n".join(chunks["project_material"]).strip(),
            risk_flags=tuple(sorted(set(risk_flags))),
        )

    def _summarize_profile(self, evidence: EvidencePack) -> str:
        if not evidence.student_profile:
            return "资料缺失：尚未读取到学生画像。请补充专业、年级、兴趣、项目、约束和目标城市。"
        return self._bullets_from_text(evidence.student_profile, limit=6)

    def _choose_direction(self, evidence: EvidencePack) -> str:
        # 仅依据画像与简历判断方向，不纳入 role_material，避免岗位资料关键词误判方向。
        text = f"{evidence.student_profile}\n{evidence.resume}"
        if "不想写代码" in text or "策展" in text or "产品经理" in text:
            return "建议优先探索产品经理或偏业务分析方向，同时谨慎评估高代码强度岗位。"
        if "AI" in text or "Agent" in text or "Python" in text:
            return "建议优先探索 AI 应用开发方向，并将后端开发作为稳定备选方向。"
        return "建议先完成资料补充，再在 AI 应用开发、后端开发和产品经理之间做最终选择。"

    def _compare_roles(self, evidence: EvidencePack) -> str:
        # 同 _choose_direction：仅用画像与简历，避免岗位资料关键词污染方向比较。
        text = f"{evidence.student_profile}\n{evidence.resume}"
        ai_note = (
            "与 AI、Python、Agent 或工具调用经验相关。"
            if any(keyword in text for keyword in ["AI", "Python", "Agent"])
            else "当前证据不足，需要补充 AI 项目或学习记录。"
        )
        backend_note = "需要持续写代码和补齐后端基础。" if "不想写代码" in text else "可作为工程能力稳健备选。"
        product_note = (
            "与策展、需求表达或低代码偏好更接近。"
            if any(keyword in text for keyword in ["策展", "不想写代码", "产品"])
            else "需要补充用户研究、PRD 和沟通推进证据。"
        )
        return f"- AI 应用开发：{ai_note}\n- 后端开发：{backend_note}\n- 产品经理：{product_note}"

    def _summarize_gaps(self, evidence: EvidencePack) -> str:
        if not evidence.student_profile and not evidence.resume:
            return "- 资料缺失：无法可靠判断能力差距。\n- 建议补充课程、项目、技能、约束和目标岗位。"
        return "- 技术/领域：围绕目标方向补齐核心技能。\n- 项目：把经历整理为可展示作品。\n- 表达：准备项目讲述、岗位匹配理由和复盘材料。"

    def _resume_advice(self, evidence: EvidencePack) -> str:
        if not evidence.resume:
            return "资料缺失：尚未读取到简历草稿。请补充项目经历、技能和成果指标。"
        return "将简历项目改写为“背景-行动-结果-证据”结构，避免泛泛描述，突出可验证交付物。"

    def _summarize_risks(self, evidence: EvidencePack) -> str:
        lines = ["文件内容均按不可信资料处理，不会覆盖系统或运行规则。"]
        if evidence.risk_flags:
            lines.append("已检测到边界风险：" + "、".join(evidence.risk_flags) + "。")
        lines.append("报告不承诺就业、录用、薪资或面试结果。")
        return "\n".join(f"- {line}" for line in lines)

    def _missing_questions(self, evidence: EvidencePack) -> str:
        questions = []
        if not evidence.student_profile:
            questions.append("学生画像")
        if not evidence.resume:
            questions.append("简历草稿")
        if not evidence.role_material:
            questions.append("岗位资料")
        base = ["英语水平", "城市偏好", "可实习时间", "目标行业", "每周可投入时间"]
        return "、".join(questions + base)

    def _bullets_from_text(self, text: str, limit: int) -> str:
        lines = []
        for raw in text.splitlines():
            cleaned = raw.strip().lstrip("- ").strip()
            if cleaned and not cleaned.startswith("#"):
                lines.append(f"- {cleaned}")
            if len(lines) >= limit:
                break
        return "\n".join(lines) if lines else "资料已读取，但缺少可结构化摘要的字段。"
