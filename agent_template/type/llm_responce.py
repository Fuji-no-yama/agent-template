from dataclasses import dataclass
from typing import Any

from .history import History


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMResponse:
    content: str
    is_tool_call: bool
    tool_name: str | None
    tool_id: str | None
    tool_args: dict[str, Any] | None
    return_history: History
