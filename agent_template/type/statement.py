from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=False, slots=True, kw_only=True)
class Statement:
    """
    会話履歴において1つの発言を表すクラス
    """

    role: Literal["user", "assistant", "system"]  # "user", "assistant", "system"
    content: str
    whose: str | None = None  # "user" or agent.name (ただしマルチエージェントの際にしか使用されない)
