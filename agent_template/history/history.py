from dataclasses import dataclass
from typing import Any


@dataclass(frozen=False, slots=True, kw_only=True)
class History:
    content: list[Any]  # チャットの履歴(LLMプロバイダによっては専用オブジェクトを入れる可能性があるのでAny)

    def add_user_message(self, content: str) -> None:
        self.content.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": content},
                ],
            },
        )

    def add_system_message(self, content: str) -> None:
        self.content.append(
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": content},
                ],
            },
        )

    def add_assistant_message(self, content: str) -> None:
        self.content.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content},
                ],
            },
        )
