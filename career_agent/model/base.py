from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    text: str
    reasoning_summary: list[str] | None = field(default=None)
    raw_model: str = ""


class LLMProvider(Protocol):
    async def complete(self, prompt: str, *, system: str = "",
                       role: str = "default") -> LLMResult:
        ...

