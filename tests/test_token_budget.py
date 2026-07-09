from career_agent.config import TokenBudgetConfig
from career_agent.runtime.token_budget import TokenBudgetManager, estimate_tokens


def test_estimate_tokens_counts_chinese_and_ascii() -> None:
    assert estimate_tokens("职业规划") >= 3
    assert estimate_tokens("career planning agent") >= 4


def test_budget_triggers_compression_by_watermark() -> None:
    manager = TokenBudgetManager(TokenBudgetConfig(max_context_tokens=100, compression_watermark=0.5))
    should, reason = manager.should_compress("中文" * 80, loaded_skill_count=1, step=1)
    assert should is True
    assert reason == "context_tokens_exceed_watermark"

