from typing import Any

from career_agent.config import Settings
from career_agent.model.bailian_provider import BailianResponsesLLM
from career_agent.model.base import LLMProvider
from career_agent.model.mock_provider import MockLLM


_DEFAULT_BAILIAN_MODEL = "qwen3.7-plus"


def build_llm_provider(
    kind: str,
    *,
    api_key: str = "",
    token: str = "",
    base_url: str = "",
    model: str = "",
    protocol: str = "openai_responses",
    enable_thinking: bool = True,
    client: Any | None = None,
) -> LLMProvider:
    normalized = _normalize_provider(kind)
    if normalized == "mock":
        return MockLLM()
    if normalized in {"bailian", "dashscope"} and protocol == "openai_responses":
        return BailianResponsesLLM(
            api_key or token,
            base_url,
            model or _DEFAULT_BAILIAN_MODEL,
            enable_thinking=enable_thinking,
            client=client,
        )
    raise ValueError(f"unsupported llm protocol: {protocol}")


def llm_from_settings(settings: Settings) -> LLMProvider:
    return build_llm_provider(
        settings.llm_provider,
        api_key=settings.llm_api_key.get_secret_value(),
        token=settings.llm_token.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        protocol=settings.llm_protocol,
        enable_thinking=settings.llm_enable_thinking,
    )


def _normalize_provider(kind: str) -> str:
    if kind in {"local", "test"}:
        return "mock"
    return kind

