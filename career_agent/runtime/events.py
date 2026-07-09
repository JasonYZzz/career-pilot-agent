from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunEvent:
    """CLI 可订阅的运行事件。

    参数: type 事件类型；message 面向人类的短消息；payload 机器可读字段。
    返回: 不适用。
    """

    type: str
    message: str
    payload: dict[str, object] = field(default_factory=dict)


EventSink = Callable[[RunEvent], None]
