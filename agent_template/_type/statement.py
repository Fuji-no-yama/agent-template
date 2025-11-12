from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=False, slots=True, kw_only=True)
class Statement:
    """
    会話履歴において1つの発言を表すクラス
    """

    role: Role  # "user", "assistant", "system"
    content: str
    whose: str | None = None  # "user" or agent.name (ただしマルチエージェントの際にしか使用されない)
