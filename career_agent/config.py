from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class TokenBudgetConfig:
    max_context_tokens: int = 12000
    compression_watermark: float = 0.75
    hard_watermark: float = 0.90
    final_answer_reserved_tokens: int = 2500
    per_tool_result_max_chars: int = 6000
    per_skill_max_tokens: int = 1600
    max_loaded_skills: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "bailian"
    llm_protocol: str = "openai_responses"
    llm_model: str = "qwen3.7-plus"
    llm_api_key: SecretStr = SecretStr("")
    llm_token: SecretStr = SecretStr("")
    llm_base_url: str = ""
    dashscope_api_key: SecretStr = SecretStr("")
    bailian_base_url: str = (
        "https://[workspace-id].cn-beijing.maas.aliyuncs.com/api/v2/apps/"
        "protocols/compatible-mode/v1"
    )
    llm_enable_thinking: bool = True
    trace_reasoning_summary: bool = False
    max_steps: int = 12
    max_context_tokens: int = 12000
    compression_watermark: float = 0.75

    @model_validator(mode="after")
    def fill_llm_defaults(self) -> "Settings":
        if not self.llm_base_url and self.llm_provider in {"bailian", "dashscope"}:
            self.llm_base_url = self.bailian_base_url
        if not self.llm_model and self.llm_provider in {"bailian", "dashscope"}:
            self.llm_model = "qwen3.7-plus"
        if not self.llm_api_key.get_secret_value() and self.dashscope_api_key.get_secret_value():
            self.llm_api_key = self.dashscope_api_key
        if (
            self.llm_provider in {"bailian", "dashscope"}
            and not self.llm_api_key.get_secret_value()
        ):
            self.llm_provider = "mock"
            self.llm_protocol = "mock"
        return self

    def token_budget(self) -> TokenBudgetConfig:
        return TokenBudgetConfig(
            max_context_tokens=self.max_context_tokens,
            compression_watermark=self.compression_watermark,
        )


def resolve_workspace(path: str) -> Path:
    return Path(path).expanduser().resolve()

