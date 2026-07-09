from career_agent.config import TokenBudgetConfig


def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars * 0.8 + other_chars / 4))


class TokenBudgetManager:
    def __init__(self, config: TokenBudgetConfig) -> None:
        self.config = config

    def should_compress(self, context: str, loaded_skill_count: int, step: int) -> tuple[bool, str]:
        tokens = estimate_tokens(context)
        if tokens >= int(self.config.max_context_tokens * self.config.compression_watermark):
            return True, "context_tokens_exceed_watermark"
        if tokens >= int(self.config.max_context_tokens * self.config.hard_watermark):
            return True, "context_tokens_exceed_hard_watermark"
        if loaded_skill_count > self.config.max_loaded_skills:
            return True, "too_many_loaded_skills"
        if step > 8:
            return True, "long_running_context"
        return False, "within_budget"
