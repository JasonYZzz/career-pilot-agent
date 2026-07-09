# CareerPilot Agent

CareerPilot Agent 是一个面向大学生职业规划的轻量命令行 AI Agent Runtime。它覆盖机试要求中的核心循环、工具调用、Skill 渐进加载、token 预算压缩、边界处理和 trace 导出，不包含前端 UI、数据库、向量库或部署系统。

## 安装

```bash
uv sync --extra dev
```

常用验证命令：

```bash
uv run pytest -q
uv run ruff check .
uv run mypy .
```

## 模型配置

默认真实模型入口使用百炼 qwen3.7-plus 的 OpenAI Responses 兼容模式。运行前需要配置有效的 `DASHSCOPE_API_KEY` 或 `LLM_API_KEY`，并把 `LLM_BASE_URL` 替换为真实百炼 workspace 地址。

```bash
DASHSCOPE_API_KEY=sk-... \
LLM_PROVIDER=bailian \
LLM_PROTOCOL=openai_responses \
LLM_MODEL=qwen3.7-plus \
LLM_BASE_URL='https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1' \
uv run career-agent llm-smoke --prompt "用一句话回复 OK"
```

配置项示例见 `.env.example`。项目保留 `LLM_PROTOCOL=openai_responses`，并通过 `career_agent.model.factory.llm_from_settings()` 创建 provider。

## 运行示例

```bash
DASHSCOPE_API_KEY=sk-... \
LLM_PROVIDER=bailian \
LLM_PROTOCOL=openai_responses \
LLM_MODEL=qwen3.7-plus \
LLM_BASE_URL='https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1' \
uv run career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md、data/injection_resume.md 和 data/job_roles 下的岗位资料，帮我在 AI应用开发、后端开发、产品经理三个方向中选择最适合的职业路径，生成 90 天行动计划，并给出每周复盘提醒草案。" \
  --workspace ./workspace \
  --trace ./trace.json
```

### CLI 运行可见性

`career-agent run` 默认输出进度事件，包括 step、token 估算、模型决策、工具调用、压缩和结束原因。若只想保留最终摘要，可加 `--quiet`。

运行结束后请查看 `workspace/outputs/run_status.json`。当 `report_generated=false` 时，本次没有生成新的 `career_plan.md`，已有报告可能是旧产物。

输出文件：

- `workspace/outputs/career_plan.md`：职业规划报告，不承诺就业、录用或薪资结果。
- `workspace/outputs/reminder_plan.json`：提醒草案，需要用户确认后才能创建真实提醒。
- `workspace/outputs/run_status.json`：本次运行状态，标记是否真的生成了新的报告。
- `trace.json`：可复盘运行 trace。

可选本地服务占位命令：

```bash
uv run career-agent serve --host 127.0.0.1 --port 8000
```

## 示例工作区

`workspace/data` 包含学生画像、简历草稿、课程、项目经历、注入样例和岗位资料。`workspace/skills/index.json` 只在启动时读取元数据，具体 markdown Skill 会在运行过程中按需加载，避免启动时把所有 Skill 放入上下文。

`examples/trace_with_compression.json` 是完整样例 trace，包含：

- `model_call`
- `tool_call`
- `skill_load`
- `token_budget`
- `compression`
- `boundary_event`

可重新生成：

```bash
DASHSCOPE_API_KEY=sk-... \
LLM_PROVIDER=bailian \
LLM_PROTOCOL=openai_responses \
LLM_MODEL=qwen3.7-plus \
LLM_BASE_URL='https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1' \
uv run career-agent run \
  --task "请根据 data/student_profile.md、data/resume_draft.md、data/injection_resume.md 和 data/job_roles 下的岗位资料，帮我在 AI应用开发、后端开发、产品经理三个方向中选择最适合的职业路径，生成 90 天行动计划，并给出每周复盘提醒草案。" \
  --workspace ./workspace \
  --trace ./examples/trace_with_compression.json
```

## 架构

运行循环是 `plan -> act -> observe -> compress -> evaluate -> final`：

- `Planner` 输出结构化 `AgentDecision`，不直接执行工具。
- `ToolRegistry` 统一注册 `list_dir`、`read_file`、`write_file`、`todo_update`、`get_time`、`create_reminder`、`restricted_shell`。
- `SkillRegistry` 只加载 `index.json`，`SkillLoader` 按任务需要渐进加载 markdown Skill。
- `TokenBudgetManager` 每轮估算 token，超过水位触发 `ContextCompressor`。
- `BoundaryGuard` 限制 workspace 路径、敏感文件、prompt injection、隐私字段、提醒确认和高风险 shell。
- `TraceLogger` 导出模型、工具、Skill、预算、压缩、边界和耗时信息。
- `ReportSynthesizer` 从工具读取结果和压缩摘要生成报告，报告必须随工作区资料变化而变化。
- Planner 对真实模型返回的决策做 allowlist 校验，未知工具、未知 Skill、越权路径和私有运行参数会被转为可追踪的澄清/边界结果。
- 压缩后仍保留新增工具观察，避免后续步骤看不到压缩之后的事实。
- 提醒工具默认只写草案；模型传入 `confirmed=true` 不会被视为用户确认。

