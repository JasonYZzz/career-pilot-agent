from typing import Any

from pydantic import SecretStr

from career_agent.config import Settings
from career_agent.model.base import LLMResult
from career_agent.model.factory import build_llm_provider, llm_from_settings


class FakeResponses:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> object:
        self.last_kwargs = kwargs
        reasoning = type(
            "Reasoning",
            (),
            {
                "type": "reasoning",
                "summary": [type("Summary", (), {"text": "简短推理摘要"})()],
            },
        )()
        message = type(
            "Message",
            (),
            {
                "type": "message",
                "content": [type("Content", (), {"text": "最终答案"})()],
            },
        )()
        return type("Response", (), {"output": [reasoning, message]})()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


async def test_bailian_responses_uses_qwen_thinking_and_extracts_message() -> None:
    fake_client = FakeClient()
    llm = build_llm_provider(
        "bailian",
        api_key="sk-test",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        model="qwen3.7-plus",
        protocol="openai_responses",
        enable_thinking=True,
        client=fake_client,
    )
    result = await llm.complete("ping", system="brief")

    assert fake_client.responses.last_kwargs is not None
    assert result.text == "最终答案"
    assert result.reasoning_summary == ["简短推理摘要"]
    assert fake_client.responses.last_kwargs["model"] == "qwen3.7-plus"
    assert fake_client.responses.last_kwargs["input"] == "ping"
    assert fake_client.responses.last_kwargs["instructions"] == "brief"
    assert fake_client.responses.last_kwargs["extra_body"] == {"enable_thinking": True}


async def test_mock_provider_is_deterministic() -> None:
    llm = build_llm_provider("mock")
    result = await llm.complete("hello")
    assert result == LLMResult(text="I found context for that.")


def test_settings_factory_uses_dashscope_key_fallback() -> None:
    settings = Settings(
        llm_provider="bailian",
        llm_protocol="openai_responses",
        llm_model="qwen3.7-plus",
        llm_api_key=SecretStr(""),
        dashscope_api_key=SecretStr("sk-from-dashscope"),
        llm_base_url="https://workspace.cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
    llm = llm_from_settings(settings)
    assert llm.__class__.__name__ == "BailianResponsesLLM"
