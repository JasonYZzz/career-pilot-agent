from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from career_agent.model.base import LLMResult


class BailianResponsesLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        enable_thinking: bool = True,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._enable_thinking = enable_thinking
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=self._base_url)

    async def complete(self, prompt: str, *, system: str = "",
                       role: str = "default") -> LLMResult:
        _ = role  # role 仅 MockLLM 使用，真实 provider 忽略
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "extra_body": {"enable_thinking": self._enable_thinking},
        }
        if system:
            kwargs["instructions"] = system
        response = await self._client.responses.create(**kwargs)
        return _extract_result(response, self._model)


def _extract_result(response: Any, model: str) -> LLMResult:
    reasoning_summary: list[str] = []
    final_parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", "")
        if item_type == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", "")
                if text:
                    reasoning_summary.append(str(text))
        elif item_type == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", "")
                if text:
                    final_parts.append(str(text))
    return LLMResult(text="".join(final_parts), reasoning_summary=reasoning_summary, raw_model=model)

