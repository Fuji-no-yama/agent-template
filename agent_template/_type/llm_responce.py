from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_template.history.history import History


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMResponse:
    content: str
    is_tool_call: bool
    tool_name: str | None
    tool_id: str | None
    tool_args: Mapping[str, Any] | None
    return_history: History | None = None
