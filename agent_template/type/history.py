from dataclasses import dataclass
from typing import Any

from .statement import Statement


@dataclass(frozen=False, slots=True, kw_only=True)
class History:
    """
    1つのエージェント内部で使用される履歴を扱うクラス(ツール実行履歴などを含んだ形で利用される)
    """

    content: list[Statement | Any]  # チャットの履歴(LLMプロバイダによっては専用オブジェクトを入れる可能性があるのでAnyもOK)

    def add_user_message(self, content: str) -> None:
        self.content.append(
            Statement(role="user", content=content),
        )

    def add_system_message(self, content: str) -> None:
        self.content.append(
            Statement(role="system", content=content),
        )

    def add_assistant_message(self, content: str) -> None:
        self.content.append(
            Statement(role="assistant", content=content),
        )

    def add_object(self, obj: Any) -> None:  # noqa: ANN401
        self.content.append(obj)

    def get_content(self) -> list[dict]:  # LLMに直接渡せる内容を取得する(将来的にはプロバイダごとに切り分ける必要?)
        ret_list = []
        for item in self.content:
            if isinstance(item, Statement):
                if item.role in ["user", "system"]:
                    content = [
                        {"type": "input_text", "text": item.content},
                    ]
                elif item.role == "assistant":
                    content = [
                        {"type": "output_text", "text": item.content},
                    ]
                ret_list.append(
                    {
                        "role": item.role,
                        "content": content,
                    },
                )
            else:  # LLMプロバイダから返ってきたオブジェクトの場合はそのまま追加
                ret_list.append(item)
        return ret_list
