# CareerPilot Agent Design Spec

## 1. 项目定位

**项目名称**：CareerPilot Agent：大学生职业规划智能体

**交付形态**：命令行优先，本地后端服务可选。不做前端 UI。

**一句话目标**：实现一个面向大学生职业规划的轻量 AI Agent Runtime。它能读取本地学生资料与岗位资料，按需调用工具和 Skill，在有限 token 预算内完成职业方向匹配、能力差距分析、简历建议、面试准备与 30/60/90 天行动计划，并导出可复盘 trace。

本项目不做前端 UI、部署、数据库、向量库或复杂多 Agent 平台。重点展示：

- Agent 核心循环：plan -> act -> observe -> compress -> evaluate -> final
- Tool Registry：文件、目录、todo、时间、提醒草案、受限 shell
- Skill Registry：职业测评、岗位匹配、能力差距、简历优化、面试规划、报告写作
- Token Budget Manager：预算估算、水位判断、上下文压缩
- Boundary Guard：路径、隐私、prompt injection、提醒确认、高风险 shell、重复失败
- Trace Logger：完整记录模型调用、工具调用、Skill 加载、压缩、边界事件和耗时

## 2. 机试原始需求固化

本项目必须直接覆盖 AI Agent 机试题的要求：

- 从 0 实现一个命令行或本地服务形式的 AI Agent。
- Agent 能接收用户任务，自主决定是否调用工具或 Skill。
- Agent 能在有限 token 预算内完成多步任务，并导出可复盘运行 trace。
- 支持多轮推理和多步任务推进，能在继续调用工具和给出最终答案之间做判断，并有明确终止条件。
- 至少支持 3 个工具；本项目 MVP 支持目录检索、文件读取、文件写入、todo 状态管理、时间获取，扩展支持提醒草案和受限 shell。
- 支持从本地目录加载 Skill，并按任务需要渐进式加载，而不是启动时全量放入上下文。
- 每轮具备 token 预算意识，有预算上限、压缩水位和压缩策略。
- 压缩后尽量保留任务目标、关键约束、当前进度、未完成事项和重要工具结果。
- 边界处理覆盖工具失败、工具超时、工具结果过大、prompt injection、隐私数据访问、长期记忆写入、未经确认的提醒创建、高风险 shell 命令、重复失败动作和预算不足。
- Trace 至少包含模型调用、工具调用、Skill 加载、token 估算、压缩触发、边界处理和耗时信息。
- README 必须包含安装方式、运行命令、模型配置方式和一个示例任务。
- 至少提供 1 份完整 trace 样例，其中包含工具调用和上下文压缩。
- 交付物需要包含简短说明：如何使用 AI 编程工具，以及架构和关键取舍判断。

## 3. 需求映射

| 机试要求 | CareerPilot Agent 对应能力 | 验收方式 |
| --- | --- | --- |
| Agent 核心循环 | Supervisor 每轮输出结构化决策，决定加载 Skill、调用 Tool、压缩或最终回答 | trace 中出现多轮 model_call 和终止原因 |
| 至少 3 个 Tool | 必做 list_dir、read_file、write_file、todo_update、get_time | trace 中至少 3 类 tool_call |
| 本地 Skill 渐进加载 | 启动只读 skills/index.json，执行中按任务阶段加载 markdown Skill | trace 中出现 skill_load，且不是启动时全量加载 |
| Token 预算与压缩 | 每轮估算 tokens，超过 75% 水位触发 compression | 样例 trace 包含 compression span |
| 边界处理 | 工具失败、超时、过大结果、注入、隐私、提醒确认、shell 风险 | trace 中记录 boundary_event |
| 可观测 Trace | 输出 trace.json，包含模型、工具、Skill、预算、压缩、边界和耗时 | examples/trace_with_compression.json 可复盘 |

## 4. 技术选型

技术选型参考 `/Users/mac/ai-project/v-rag` 的后端工程风格，但做机试场景裁剪。

| 维度 | 选择 | 说明 |
| --- | --- | --- |
| 语言 | Python `>=3.12` | 对齐 v-rag 后端风格，便于 AI/Agent 工程实现 |
| 包管理 | `uv` | 用 `uv sync --extra dev` 固定依赖和测试环境 |
| CLI | Typer | 提供 `career-agent run`，便于评审复现 |
| 本地服务 | FastAPI + uvicorn，可选 | 只作为后端 API 入口，不做前端 UI |
| 配置 | pydantic-settings | 使用 `.env` 管理模型、预算、运行参数 |
| 模型接口 | 百炼 qwen3.7-plus Responses 兼容模式 | 默认按用户提供的 OpenAI SDK `client.responses.create(...)` 代码路径设计 |
| LLM 集成形态 | 参考 v-rag provider/factory | async `LLMProvider` 协议、`build_llm_provider()`、`llm_from_settings()`、mock provider、百炼 Responses 解析 |
| SDK/HTTP | OpenAI SDK + provider tests | 默认真实调用使用 `openai` SDK；测试使用 fake client 验证请求参数、输出解析和错误路径 |
| 离线演示 | Mock provider | 无 API key 时仍可生成报告和 trace 样例 |
| 测试 | pytest + pytest-asyncio | 覆盖工具、Skill、预算、压缩、边界、CLI smoke 和模型 provider |
| 质量 | ruff + mypy | 对齐 v-rag 后端质量门槛 |
| 可观测 | 自研 trace schema | 不引入 Langfuse/OTel 作为 MVP 依赖，trace JSON 是交付物 |

默认模型配置：

```env
LLM_PROVIDER=bailian
LLM_PROTOCOL=openai_responses
LLM_MODEL=qwen3.7-plus
LLM_API_KEY=
LLM_TOKEN=
LLM_BASE_URL=https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1
DASHSCOPE_API_KEY=your_api_key
LLM_ENABLE_THINKING=true
TRACE_REASONING_SUMMARY=false
```

默认请求形态：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
)

response = client.responses.create(
    model="qwen3.7-plus",
    input="...",
    extra_body={"enable_thinking": True},
)
```

实现约束：

- `BailianResponsesLLM` 是默认真实模型 provider。
- `MockLLM` 是默认测试和无 key 演示 provider。
- `career-agent llm-smoke --prompt "用一句话回复 OK"` 用于在跑完整 Agent 前验证真实大模型通路。
- `Settings` 参考 v-rag 的 fallback 方式：`LLM_API_KEY` 为空时使用 `DASHSCOPE_API_KEY`，`LLM_BASE_URL` 为空时要求用户显式配置百炼 workspace base URL。
- Provider 解析 `response.output`：`reasoning` 类型只作为可选 trace 摘要，`message` 类型才作为最终文本交给 Planner。
- `enable_thinking` 默认开启，但 trace 默认不保存完整推理过程；如需复盘只保存截断 summary。
- Planner 输出仍使用项目内 `AgentDecision` JSON schema，不能让模型直接执行工具。
- 本项目工具调用由 Runtime 执行并记录 trace，不依赖 OpenAI 托管工具来替代本地 Tool Registry。

## 5. 推荐运行入口

```bash
career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md 和 data/job_roles 下的岗位资料，帮我在 AI应用开发、后端开发、产品经理三个方向中选择最适合的职业路径，并生成 90 天行动计划。如果需要提醒，请只生成提醒草案，不要直接创建。" \
  --workspace ./workspace \
  --trace ./trace.json
```

可选本地服务：

```bash
career-agent serve --host 127.0.0.1 --port 8000
```

## 6. Workspace 约定

```text
workspace/
  data/
    student_profile.md
    resume_draft.md
    course_transcript.md
    project_experience.md
    injection_resume.md
    job_roles/
      backend_engineer.md
      ai_application_engineer.md
      product_manager.md
      data_analyst.md
      all_roles_long.md
  skills/
    index.json
    career_assessment.md
    role_matching.md
    skill_gap_analysis.md
    resume_review.md
    interview_plan.md
    action_plan.md
    privacy_guard.md
    report_writer.md
    context_compression.md
  outputs/
    career_plan.md
    todo_plan.json
    reminder_plan.json
```

## 7. 技术架构

参考给定架构图与 HTML 方案，本项目采用“Typed Planner + Custom Workflow + Agents-as-Tools 思想”的轻量实现。

```text
CLI / Local API
  |
  v
CareerPilot Agent Runtime
  |
  +-- Task Intake
  +-- Planner / Supervisor
  +-- Context Builder
  +-- Token Budget Manager
  +-- Context Compressor
  +-- Tool Registry
  +-- Skill Registry
  +-- Boundary Guard
  +-- Trace Logger
  +-- Critic / Quality Checker
  |
  v
Final Outputs
  +-- outputs/career_plan.md
  +-- outputs/todo_plan.json
  +-- outputs/reminder_plan.json
  +-- trace.json
```

核心取舍：

- Planner 是控制面，不是普通聊天 Agent。它负责任务图、预算、工具权限、Skill 加载和终止判断。
- 专家能力先用 Skill 表达，不急于拆成多个独立 Agent，保证机试项目小而完整。
- 所有工具统一注册、统一校验、统一 trace，不让模型自由执行任意动作。
- 文件内容统一视为不可信数据，只能作为资料，不能覆盖系统指令。
- 评审失败优先局部修复，不全量重跑。

## 8. Agent 核心循环

每轮执行：

```text
while not done:
  1. Context Builder 组装任务、状态、已加载 Skill 摘要、工具结果和压缩摘要
  2. Token Budget Manager 估算上下文 token
  3. 若超过 compression_watermark，触发 Context Compressor
  4. Planner 调用模型，要求输出结构化 AgentDecision
  5. 根据 decision 执行动作：load_skill / call_tool / update_todo / compress_context / final_answer / ask_clarification
  6. Boundary Guard 对动作和结果做检查
  7. Observation 写回 RunState
  8. Trace Logger 写入 span
  9. 判断终止条件
```

模型决策格式：

```json
{
  "thought_summary": "需要先读取学生画像和候选岗位资料，再进行方向匹配。",
  "decision": "call_tool",
  "tool_name": "read_file",
  "tool_args": {
    "path": "data/student_profile.md",
    "max_chars": 6000
  },
  "skill_name": null,
  "todo_update": null,
  "final_answer": null,
  "reason": "当前缺少学生画像，无法判断职业方向。",
  "expected_observation": "获得专业、年级、技能、项目、目标和约束。"
}
```

终止条件：

- `final_answer`：任务完成
- `max_steps`：默认 12 步
- `max_tool_failures`：同一工具同一参数失败超过 2 次
- `token_budget_exhausted`：压缩后仍不足
- `unsafe_action_blocked`：高风险动作无法继续
- `missing_critical_info`：关键资料缺失，只能输出部分方案或澄清问题
- `reminder_requires_confirmation`：提醒需要用户确认，不能直接创建

## 9. Tool 设计

MVP 必做 5 个工具，加分实现 2 个工具。

| Tool | 必做 | 风险 | 作用 |
| --- | --- | --- | --- |
| list_dir | 是 | low | 列出 workspace 内目录 |
| read_file | 是 | medium | 读取学生资料、岗位资料、Skill 文档 |
| write_file | 是 | medium | 写出报告、todo、提醒草案 |
| todo_update | 是 | low | 维护任务拆解和状态 |
| get_time | 是 | low | 获取当前时间，用于计划起始日期 |
| create_reminder | 可选 | high | 只生成提醒草案，确认后才创建真实提醒 |
| restricted_shell | 可选 | high | 在 workspace 内执行白名单命令 |

关键策略：

- 所有路径必须限制在 workspace 内。
- `read_file` 禁止读取 `.env`、secret、key、token 等敏感文件。
- `write_file` 默认只能写入 `outputs/`。
- 单次工具结果默认最多 6000 chars，超过则截断、摘要并记录 trace。
- `restricted_shell` 只允许 `pwd`、`ls`、`cat`、`head`、`wc`、`python -m pytest`。
- shell 禁止 `rm`、`mv`、`curl`、`wget`、`ssh`、`sudo`、`chmod`、`chown`、`env`，以及包含 `&&`、`|`、`;`、反引号、`$()` 的复杂命令。

## 10. Skill 设计

启动时只读取 `workspace/skills/index.json`，后续按任务需要渐进加载具体 Skill。

推荐 Skill：

| Skill | 触发场景 | 预算 |
| --- | --- | --- |
| career_assessment | 学生画像、兴趣、专业、方向选择 | 1200 tokens |
| role_matching | 后端、AI 应用、产品、数据分析等岗位匹配 | 1600 tokens |
| skill_gap_analysis | 能力差距、学习路线、补齐路径 | 1400 tokens |
| resume_review | 简历、项目经历、简历优化 | 1400 tokens |
| interview_plan | 面试、笔试、求职准备 | 1200 tokens |
| action_plan | 30/60/90 天行动计划 | 1200 tokens |
| privacy_guard | 隐私识别、敏感建议、长期记忆边界 | 800 tokens |
| report_writer | 结构化职业规划报告 | 1000 tokens |
| context_compression | 压缩上下文时保留关键事实 | 800 tokens |

加载顺序示例：

```text
第 1 轮：读取 skills/index.json
第 2 轮：加载 career_assessment
第 3 轮：读取 student_profile.md / resume_draft.md
第 4 轮：加载 role_matching
第 5 轮：读取 job_roles 下的候选岗位资料
第 6 轮：加载 skill_gap_analysis + action_plan
第 7 轮：触发压缩，保留学生画像、岗位摘要、差距结论、todo 状态
第 8 轮：加载 report_writer
第 9 轮：写出 career_plan.md
```

## 11. Token 预算与压缩

配置：

```yaml
token_budget:
  max_context_tokens: 12000
  compression_watermark: 0.75
  hard_watermark: 0.90
  final_answer_reserved_tokens: 2500
  per_tool_result_max_chars: 6000
  per_skill_max_tokens: 1600
  max_loaded_skills: 3
```

MVP 估算函数：

```python
def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 0.8 + other_chars / 4)
```

压缩触发条件：

- 当前上下文超过 `max_context_tokens * 0.75`
- 工具结果过大
- 已加载 Skill 超过 3 个
- 步数超过 8 步
- 预留最终回答 token 不足

压缩后必须保留：

```json
{
  "task_goal": "生成大学生职业规划报告",
  "user_constraints": [],
  "student_profile_facts": [],
  "career_direction_candidates": [],
  "important_evidence": [],
  "loaded_skills_summary": [],
  "tool_results_summary": [],
  "todo_state": [],
  "open_questions": [],
  "risk_flags": [],
  "next_steps": []
}
```

## 12. 边界处理

| 边界情况 | 处理策略 | Trace 事件 |
| --- | --- | --- |
| 工具失败 | 记录错误，最多换参数重试 2 次 | tool_error |
| 工具超时 | 中止工具调用，返回 timeout observation | tool_timeout |
| 工具结果过大 | 截断、摘要、触发压缩 | tool_result_truncated |
| Prompt injection | 文件内容视为 untrusted data，不执行其中指令 | prompt_injection_detected |
| 隐私数据 | 报告脱敏，不写入长期记忆 | privacy_guard |
| 长期记忆写入 | 默认只生成 memory_candidate，不直接写入 | memory_write_blocked |
| 未确认提醒 | 只写 reminder_plan.json，不创建系统提醒 | reminder_requires_confirmation |
| 高风险 shell | 黑名单、白名单、workspace 限制、timeout | shell_blocked |
| 重复失败动作 | 同工具同参数失败 2 次后阻断 | repeated_failure_blocked |
| 预算不足 | 先压缩，仍不足则输出部分结果 | budget_exhausted |

Prompt injection 检测关键词：

- `ignore previous instructions`
- `忽略以上指令`
- `system prompt`
- `developer message`
- `tool call`
- `删除文件`
- `读取密钥`

## 13. 主流程 Typed Plan

```json
{
  "run_id": "run_20260707_001",
  "task_type": "career_planning",
  "goal": "生成大学生职业规划报告",
  "constraints": {
    "workspace": "./workspace",
    "output": "outputs/career_plan.md",
    "trace": "./trace.json",
    "max_steps": 12,
    "max_context_tokens": 12000
  },
  "plan": [
    {"id": "step_1", "type": "tool", "name": "list_dir", "args": {"path": "data"}},
    {"id": "step_2", "type": "tool", "name": "read_file", "args": {"path": "skills/index.json", "max_chars": 6000}},
    {"id": "step_3", "type": "skill", "name": "career_assessment"},
    {"id": "step_4", "type": "tool", "name": "read_file", "args": {"path": "data/student_profile.md", "max_chars": 6000}},
    {"id": "step_5", "type": "tool", "name": "read_file", "args": {"path": "data/resume_draft.md", "max_chars": 6000}},
    {"id": "step_6", "type": "skill", "name": "role_matching"},
    {"id": "step_7", "type": "tool", "name": "read_file", "args": {"path": "data/job_roles/ai_application_engineer.md", "max_chars": 6000}},
    {"id": "step_8", "type": "tool", "name": "read_file", "args": {"path": "data/job_roles/backend_engineer.md", "max_chars": 6000}},
    {"id": "step_9", "type": "tool", "name": "read_file", "args": {"path": "data/job_roles/product_manager.md", "max_chars": 6000}},
    {"id": "step_10", "type": "skill", "name": "skill_gap_analysis"},
    {"id": "step_11", "type": "skill", "name": "action_plan"},
    {"id": "step_12", "type": "tool", "name": "get_time", "args": {}},
    {"id": "step_13", "type": "compression", "name": "compress_context"},
    {"id": "step_14", "type": "skill", "name": "report_writer"},
    {"id": "step_15", "type": "tool", "name": "write_file", "args": {"path": "outputs/career_plan.md", "mode": "overwrite"}}
  ]
}
```

说明：Typed Plan 是 Planner 的初始计划，真实运行时仍允许 Agent 根据 observation 调整顺序、补读文件、加载新 Skill 或提前终止。

## 14. Trace Schema

顶层结构：

```json
{
  "run_id": "run_20260707_001",
  "task": "用户原始任务",
  "workspace": "./workspace",
  "started_at": "2026-07-07T18:30:00+08:00",
  "ended_at": "2026-07-07T18:30:42+08:00",
  "status": "success",
  "model": {
    "provider": "openai",
    "api_mode": "responses",
    "name": "qwen3.7-plus"
  },
  "config": {
    "max_steps": 12,
    "max_context_tokens": 12000,
    "compression_watermark": 0.75
  },
  "spans": [],
  "summary": {
    "model_calls": 6,
    "tool_calls": 8,
    "skill_loads": 4,
    "compressions": 1,
    "boundary_events": 2,
    "total_estimated_tokens": 21400,
    "output_files": [
      "outputs/career_plan.md",
      "outputs/todo_plan.json",
      "outputs/reminder_plan.json"
    ]
  }
}
```

必须支持的 span：

- `run_start`
- `model_call`
- `tool_call`
- `skill_load`
- `token_budget`
- `compression`
- `boundary_event`
- `todo_update`
- `final_answer`
- `run_end`

压缩 span 示例：

```json
{
  "type": "compression",
  "span_id": "span_011",
  "timestamp": "2026-07-07T18:30:23+08:00",
  "trigger": "context_tokens_exceed_watermark",
  "before_tokens": 10280,
  "after_tokens": 5420,
  "watermark": 0.75,
  "preserved": [
    "task_goal",
    "student_profile_summary",
    "career_candidates",
    "skill_gap_findings",
    "todo_state",
    "important_tool_results"
  ],
  "elapsed_ms": 840
}
```

## 15. 输出报告结构

`workspace/outputs/career_plan.md`

```markdown
# 大学生职业规划报告

## 1. 结论摘要
## 2. 学生画像摘要
## 3. 当前核心问题
## 4. 候选职业方向
### 方向一：AI 应用开发工程师
### 方向二：后端开发工程师
### 方向三：产品经理
## 5. 推荐主方向与备选方向
## 6. 能力差距分析
## 7. 简历优化建议
## 8. 面试准备计划
## 9. 30 / 60 / 90 天行动计划
## 10. 每周 Todo
## 11. 风险提醒
## 12. 需要用户进一步确认的信息
```

报告约束：

- 不承诺就业、录用、薪资结果。
- 不输出完整手机号、身份证号、密钥或其他敏感原文。
- 对每个建议给出依据、风险和下一步。
- 如果资料缺失，明确标注假设和待确认问题。

## 16. 代码模块计划

推荐 Python `>=3.12` 实现，包管理与命令执行使用 `uv`。

```text
career-agent/
  README.md
  pyproject.toml
  .env.example
  career_agent/
    __init__.py
    cli.py
    config.py
    runtime/
      agent_loop.py
      run_state.py
      planner.py
      context_builder.py
      token_budget.py
      compressor.py
      trace_logger.py
      boundary_guard.py
      critic.py
    model/
      base.py
      factory.py
      bailian_provider.py
      mock_provider.py
    tools/
      base.py
      registry.py
      file_tools.py
      todo_tool.py
      time_tool.py
      reminder_tool.py
      shell_tool.py
    skills/
      registry.py
      loader.py
      selector.py
    prompts/
      system_prompt.md
      planner_prompt.md
      compression_prompt.md
      critic_prompt.md
  workspace/
  examples/
    trace_with_compression.json
  tests/
```

核心类：

```python
@dataclass
class RunState:
    run_id: str
    task: str
    workspace: Path
    step: int = 0
    max_steps: int = 12
    messages: list = field(default_factory=list)
    loaded_skills: dict = field(default_factory=dict)
    tool_results: list = field(default_factory=list)
    todos: list = field(default_factory=list)
    compressed_summary: str | None = None
    boundary_events: list = field(default_factory=list)
    done: bool = False
    final_answer: str | None = None


@dataclass
class AgentDecision:
    decision: Literal[
        "call_tool",
        "load_skill",
        "update_todo",
        "compress_context",
        "final_answer",
        "ask_clarification",
    ]
    reason: str
    tool_name: str | None = None
    tool_args: dict | None = None
    skill_name: str | None = None
    final_answer: str | None = None


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None
    truncated: bool = False
    elapsed_ms: int = 0
```

## 17. 实现里程碑

### Phase 1：可运行闭环

目标：一个命令可跑通并生成报告和 trace。

- CLI：`career-agent run`
- 可选本地服务：`career-agent serve`
- Bailian qwen3.7-plus Responses provider + provider factory + `llm-smoke`
- Mock model：无 API key 时也能演示
- RunState / AgentDecision / ToolResult
- list_dir、read_file、write_file、todo_update、get_time
- skills/index.json 和 3 个基础 Skill
- token 估算与 trace 输出

验收：

- 能读取 workspace 示例资料
- 能写出 `outputs/career_plan.md`
- trace 中有 model_call、tool_call、skill_load、token_budget、run_end

### Phase 2：压缩与边界

目标：满足机试对压缩和边界处理的显性要求。

- Context Compressor
- `all_roles_long.md` 触发工具结果过大和压缩
- `injection_resume.md` 触发 prompt injection 边界事件
- 隐私脱敏
- reminder_plan 草案
- restricted_shell 白名单与黑名单

验收：

- `examples/trace_with_compression.json` 包含 compression
- trace 至少包含 2 个 boundary_event
- 高风险 shell 被拦截
- 未确认提醒不真实创建

### Phase 3：质量与复盘

目标：让项目看起来像真正的 Agent Runtime，而不是脚本拼装。

- Critic / Quality Checker
- 局部修复策略
- README 完整说明安装、配置、运行、模型配置和示例任务
- 架构与关键取舍说明
- 单元测试覆盖核心 Guard 与 Tool

验收：

- `pytest` 通过
- README 有一条可复制运行命令
- 示例 trace 可用于复盘 Planner 决策
- 最终报告没有隐私泄露和过度承诺

## 18. README 必须包含

- 项目简介
- 安装方式
- 模型配置方式
- Mock model 演示方式
- 运行命令
- Workspace 示例结构
- 输出文件说明
- Trace 样例说明
- 架构取舍说明
- AI 编程工具使用说明

AI 编程工具使用说明建议写法：

```text
本项目使用 AI 编程工具辅助生成初始模块骨架、Tool Schema、测试样例和 README 表达；
核心架构、边界策略、Token 压缩策略、Trace Schema 和职业规划任务映射由本人设计和裁剪。
实现过程中没有把业务问题直接交给一个通用 Chatbot，而是构建了显式 Agent Loop、
Tool Registry、Skill Registry、Budget Manager、Boundary Guard 和 Trace Logger。
```

## 19. 样例 Trace 设计

为确保样例 trace 覆盖机试要求，示例任务应包含：

- 读取目录
- 读取学生资料
- 读取简历草稿
- 读取岗位资料
- 加载至少 3 个 Skill
- `all_roles_long.md` 触发压缩
- `injection_resume.md` 触发 prompt injection 边界
- 请求“每周复盘提醒”触发 reminder_requires_confirmation
- 写出报告

推荐示例任务：

```bash
career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md、data/injection_resume.md 和 data/job_roles 下的岗位资料，帮我在 AI应用开发、后端开发、产品经理三个方向中选择最适合的职业路径，生成 90 天行动计划，并给出每周复盘提醒草案。" \
  --workspace ./workspace \
  --trace ./trace.json
```

## 20. 最小验收清单

- [ ] `career-agent run` 可执行
- [ ] 默认真实模型客户端使用 OpenAI Responses API `/v1/responses` 模式
- [ ] `career-agent llm-smoke` 可在配置 API key 后独立验证真实大模型调用
- [ ] 无 API key 时可通过 mock provider 复现
- [ ] 至少 5 个工具可用
- [ ] 工具有 schema、风险等级、timeout 和错误处理
- [ ] Skill 从本地目录按需加载
- [ ] 每轮有 token 估算
- [ ] 超过水位会触发压缩
- [ ] 压缩保留任务目标、关键约束、进度、未完成事项和重要工具结果
- [ ] prompt injection 被标记为不可信内容
- [ ] 隐私内容在报告中脱敏
- [ ] 提醒创建需要确认，只输出草案
- [ ] 高风险 shell 被拦截
- [ ] 重复失败动作被阻断
- [ ] trace 包含模型调用、工具调用、Skill 加载、token 估算、压缩、边界事件和耗时
- [ ] README 包含安装、运行、模型配置、示例任务、架构取舍和 AI 编程工具说明
- [ ] examples/trace_with_compression.json 是完整样例

## 21. 最终交付物

```text
career-agent.zip 或代码仓库
  README.md
  pyproject.toml
  .env.example
  career_agent/
  workspace/
    data/
    skills/
    outputs/
  examples/
    trace_with_compression.json
  tests/
```

这份 spec 的主线是：用职业规划这个真实业务场景承载机试要求，但实现保持小型、可复现、可观测。重点不是“生成一份职业建议”，而是展示一个能规划、能调用工具、能加载 Skill、能控预算、能处理边界、能导出 trace 的完整 Agent。
