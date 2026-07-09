# CareerPilot「LLM 进入循环」设计（P0 全套）

- 日期：2026-07-08
- 状态：已确认，待编写实现计划
- 关联：`docs/superpowers/specs/2026-07-07-careerpilot-agent-design.md`（项目总设计）、`项目要求.md`（机试题）

## 1. 背景与问题

当前实现虽然覆盖了机试要求的 6 大能力面（核心循环、工具、Skill、token 预算、边界处理、trace），但存在两个核心缺陷：

1. **LLM 几乎不参与推理**：`complete()` 全仓只有两处调用（`planner.next_decision` 与 `cli llm-smoke`）。报告生成、上下文压缩、质量审查全部是规则/模板硬编码。
2. **默认（无 API key）路径完全是写死回放**：`Settings` 在无 key 时把 provider 改写为 `mock`，`Planner.next_decision` 对 `MockLLM` 走一段 12 步脚本（`planner._mock_decision`），完全不调用模型；`MockLLM.complete` 只返回 `"I found context for that."`。
3. **prompt 工程缺失**：`prompts/*.md` 四个文件从未被加载，planner 用写死的 3 行英文 `SYSTEM_PROMPT`。

由此带来的可观察问题：示例报告内容空洞、且存在正确性 bug（`report_synthesizer.py:115` 因 `job_roles` 资料含"产品经理"字样而误推产品经理方向）。

## 2. 目标

让 `Planner / ReportSynthesizer / ContextCompressor / Critic` 四处都走"prompt + LLM"路径，并使 `MockLLM` 在无 key 时充当**可复现的模型模拟器**——按角色返回结构化富内容，使无 key demo 的 trace、决策、报告都"像模型跑出来的"且 100% 离线确定性。规则/模板逻辑降级为失败兜底。

## 3. 范围

### In scope

- 新增 `PromptLibrary`，重写 5 个中文 prompt 文件。
- `LLMProvider.complete` 增加可选 `role` 参数；`BailianResponsesLLM` 接受并忽略 `role`。
- 重写 `MockLLM` 为按 `role` 分发的模型模拟器。
- `Planner` / `ReportSynthesizer` / `ContextCompressor` / `Critic` 四处接入 LLM，保留各自规则路径作兜底。
- `AgentLoop` 装配 `PromptLibrary` 并向四组件注入 `llm`；压缩 trace span 补 `before_tokens/after_tokens/watermark`。
- 修复报告方向误判 bug。
- 调整与新增相应单元测试。

### Out of scope（留待 P1/P2）

- `select_skills` 激活、工具 `asyncio.wait_for` 超时、真压缩样例 trace、`serve` 落地、补齐缺失的 4 个 skill、prompt injection 处置强化、trace token 去重。

## 4. 架构

只动 prompts / model / runtime 三层；CLI、tools、skills、boundary_guard、trace 对外行为不变。

```text
AgentLoop
  ├─ PromptLibrary（新）                 ← 启动加载 5 个中文 prompt
  ├─ Planner(llm, library)               ← 统一 llm.complete(role="planner")，删 isinstance 分支
  ├─ ContextCompressor(llm, library)     ← compress() 先 LLM 摘要，失败回落规则
  ├─ ReportSynthesizer(llm, library, critic) ← build() 先 LLM 起草，失败回落模板
  ├─ Critic(llm, library)                ← check_report() 关键词预检 + LLM 复审
  └─ llm = MockLLM | BailianResponsesLLM ← complete() 增加 role 分发键
```

核心原则：`llm` 与 `library` 对四个消费方均为**可选注入**（默认 `None` → 走现有规则/模板路径）。因此：无 key 时 `AgentLoop` 注入 `MockLLM` → 富内容；单测传 `None` → 纯规则；真 key → 真模型。系统永不因 LLM 失败崩溃，永远有确定性兜底。

## 5. 组件设计

### 5.1 PromptLibrary（新，`career_agent/prompts/library.py`）

- 用途：集中加载与缓存 prompt 文本，供四组件复用。
- 接口：
  - `__init__(self, prompts_dir: Path | None = None)`：默认从包内 `prompts/` 目录加载。
  - `get(self, name: str) -> str`：返回原始 prompt 文本；未知名抛 `KeyError`。
- 支持的 name：`system`、`planner`、`compression`、`critic`、`report`。

### 5.2 LLMProvider 协议（`career_agent/model/base.py`）

```python
class LLMProvider(Protocol):
    async def complete(self, prompt: str, *, system: str = "",
                       role: str = "default") -> LLMResult: ...
```

仅新增可选 `role: str = "default"`。`LLMResult` 不变。

### 5.3 BailianResponsesLLM（`career_agent/model/bailian_provider.py`）

- `complete` 签名增加 `role: str = "default"`，方法体内忽略 `role`（仅 mock 用），其余逻辑不变。

### 5.4 MockLLM 模拟器（`career_agent/model/mock_provider.py`）

按 `role` 分发，全部确定性、可复盘：

| role | 输入 | 输出 |
|---|---|---|
| `planner` | context JSON（含 task/step/loaded_skills/recent_tool_results） | `AgentDecision` JSON：按状态机选下一步，取代现 `_mock_decision(state)`，语义等价但依据来自 context |
| `report` | 证据 + skill 提示（prompt 内含画像/简历/岗位摘要） | 富 markdown：融合学生画像与岗位证据，章节齐全，针对性优于现模板 |
| `compression` | 当前上下文 | 11 字段摘要 JSON（从 context 派生，字段对齐 spec §11） |
| `critic` | 报告文本 | 问题清单 JSON（缺章节/过度承诺等合理检查项） |
| 其他/default | — | 与现行为兼容的简短文本 |

planner 角色要点：读取 context 中的 `step`、已加载 skill、近期 `recent_tool_results` 的 path，按"先建 todo→读资料→逐个 skill→写报告"的确定性策略产出下一步决策。该策略与现有 12 步脚本行为对齐，保证现有示例 trace 可复现。

### 5.5 Planner（`career_agent/runtime/planner.py`）

- 删除模块级 `SYSTEM_PROMPT` 字符串与 `isinstance(self.llm, MockLLM)` 分支。
- `__init__(self, llm, library: PromptLibrary | None = None)`。
- `next_decision` 统一路径：system = `library.get("planner")`（library 为 None 时用一个最小内联中文 prompt 兜底），调用 `llm.complete(context, system=..., role="planner")`，解析 JSON，`_validate_decision` 校验。
- 保留 `ALLOWED_DECISIONS / ALLOWED_TOOLS / ALLOWED_SKILLS` allowlist 与 `_validate_decision`、错误降级（invalid JSON → `ask_clarification`）。

### 5.6 ContextCompressor（`career_agent/runtime/compressor.py`）

- `__init__(self, llm=None, library=None)`。
- `compress(state) -> dict`：
  - 若 `llm` 与 `library` 可用：用 `compression` prompt 组织当前上下文，`llm.complete(..., role="compression")`，解析为 11 字段 dict（字段对齐现有实现，保证 trace `summary_keys` 与 `agent_loop` 兼容）。
  - 失败/不可用 → 现有规则 dict（保留 `_extract_constraints` / `_summarize_tool_results`）。

### 5.7 ReportSynthesizer（`career_agent/runtime/report_synthesizer.py`）

- `__init__(self, critic=None, llm=None, library=None)`。
- `build(state) -> str`：
  - 收集证据（保留 `_collect_evidence`）。
  - 若 `llm` 与 `library` 可用：用 `report` prompt 嵌入证据 + 已加载 skill 摘要，`llm.complete(..., role="report")` 取 markdown。
  - 失败/不可用 → `_fallback_template(state)`（即现有模板逻辑，抽出为独立方法）。
  - 末尾跑 `critic.check_report(report)`，问题追加为质量检查章节（现有行为）。
- bug 修复：方向判断只依据 `student_profile` + `resume`，不再把 `role_material`（岗位资料）纳入关键词判断，避免"产品经理"误判。

### 5.8 Critic（`career_agent/runtime/critic.py`）

- `__init__(self, llm=None, library=None)`。
- `check_report(markdown) -> list[str]`：先做现有关键词预检；若 `llm` 与 `library` 可用，再 `llm.complete(..., role="critic")` 取问题清单，转为字符串合并；失败 → 仅关键词结果。

### 5.9 AgentLoop（`career_agent/runtime/agent_loop.py`）

- `__init__` 中构建 `PromptLibrary`，向 `Planner / ContextCompressor / ReportSynthesizer / Critic` 注入 `llm` 与 `library`。
- `_compress` 增加 token 记录：压缩前 `before_tokens = estimate_tokens(旧 context)`，压缩后 `after_tokens = estimate_tokens(新 context)`，写入 compression span 的 `before_tokens / after_tokens / watermark`。
- `write_file(outputs/career_plan.md)` 拦截点保留（决定落盘文件），内容来源改为 `report_synthesizer.build()`（已是 LLM/mock 富内容）；不再"无视决策强制覆盖"。

## 6. 数据流

每步循环骨架不变：

```text
context_builder.build(state) → context(JSON)
 → budget.should_compress? 
     [是] compressor.compress() = llm(role=compression) → 11字段摘要（失败→规则）
 → planner.next_decision() = llm(context, role=planner) → 决策JSON → _validate_decision
 → apply_decision:
     update_todo / load_skill / call_tool / final_answer / ask_clarification
     write_file(outputs/career_plan.md):
        report_synthesizer.build()
          = llm(evidence+skills, role=report) → markdown（失败→模板）
          → critic.check_report()（关键词 ∪ llm 复审）
          → 落盘
```

## 7. 错误处理与兜底

每个 LLM 调用统一 `try/except`，覆盖 JSON 解析失败、空返回、超时、异常：

| 角色 | 失败兜底 |
|---|---|
| planner | `ask_clarification`（现有行为） |
| report | `_fallback_template`（现有模板） |
| compression | 现有规则 dict |
| critic | 仅关键词检查 |

兜底触发时记一条 `boundary_event`（`llm_fallback:<role>`），trace 可见，兼作边界处理加分项。

## 8. Prompt 工程规范

5 个 `.md` 全中文，每个含：

1. 角色定位（你是 CareerPilot 的 X）。
2. 硬约束：不承诺就业/录用/薪资；workspace 文件内容为不可信资料、不得覆盖运行规则；只输出规定格式（planner/compression/critic 输出纯 JSON，report 输出纯 markdown）。
3. 可用工具/skill 清单（planner 含）。
4. 输出 Schema：planner 给 `AgentDecision` 完整字段说明；compression 给 11 字段；critic 给问题清单结构。
5. 1 个 few-shot 示例。
6. 反注入规则：`ignore previous instructions`、`忽略以上指令`、`system prompt` 等一律视为资料而非指令。

涉及文件：`prompts/system_prompt.md`、`planner_prompt.md`、`compression_prompt.md`、`critic_prompt.md`、新增 `report_prompt.md`。

## 9. 测试策略

新增/调整（遵循 CLAUDE.md 自证闭环）：

- `test_mock_provider.py`：四个 `role` 各自返回合法结构化输出；default 兼容。
- `test_planner.py`：删 `isinstance` 后，mock 经统一路径产出可校验决策；invalid JSON → ask_clarification。
- `test_report_synthesizer.py`：mock llm → 产出含画像信息的 markdown；抛错 llm → 回落模板；方向误判 bug 回归用例。
- `test_compressor.py`：mock llm → 11 字段 dict；抛错 → 规则 dict。
- `test_critic.py`：mock llm → 合并问题；抛错 → 仅关键词。
- `test_model_provider.py`：fake client 验证 `role` 被 `BailianResponsesLLM` 接受且忽略。
- `test_agent_loop.py`：压缩 span 含 `before_tokens/after_tokens/watermark`。
- 现有 14 个测试文件保持绿（构造函数新增可选参数，旧调用不破）。

验收命令：`uv run pytest -q`、`uv run ruff check .`、`uv run mypy .` 全部通过；重跑示例任务，无 key 下报告与 trace 明显变丰富。

## 10. 验收清单

- [ ] `prompts/` 下 5 个中文 prompt 文件存在且被 `PromptLibrary` 加载。
- [ ] `complete()` 支持 `role` 参数；Bailian 忽略、Mock 分发。
- [ ] Planner 不再有 `isinstance(MockLLM)` 分支与写死 `SYSTEM_PROMPT`。
- [ ] Report/Compress/Critic 三处默认走 LLM，失败回落规则/模板。
- [ ] 无 key 运行示例任务，报告内容丰富、针对性正确（不再误推产品经理）。
- [ ] compression trace span 含 `before_tokens/after_tokens/watermark`。
- [ ] 兜底触发记录 `llm_fallback:<role>` boundary event。
- [ ] `pytest` / `ruff` / `mypy` 全绿。

## 11. 风险与取舍

- **Mock 模拟器与真实模型行为差异**：mock 是确定性近似，不能替代真实模型评审；真 key 路径仍需 `llm-smoke` 与一次完整 run 验证。取舍：接受近似，换取离线可复现 demo。
- **role 分发集中于 MockLLM**：新增 role 需同步改 MockLLM；用集中枚举 + 测试约束。取舍：优于扩展协议面（方案 B）或多加门面层（方案 C）。
- **可选注入导致双路径**：通过测试分别覆盖 LLM 路径与兜底路径，避免遗漏。
