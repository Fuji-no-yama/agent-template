from dataclasses import dataclass

from .history import History
from .statement import Statement


@dataclass(frozen=False, slots=True, kw_only=True)
class SessionHistory(History):
    """
    セッションとして利用される履歴を扱うクラス。
    そのままHistoryとしても利用できるが、マルチエージェントでの実行を見越して以下の機能を拡張
    - 会話以外の履歴(ツール呼び出し履歴など)を削除する機能(1人のエージェントが使用後に削除して次のエージェントに渡す)
    - 立場を自動で変換してエージェントにユーザとの対話だと思わせる機能
    """

    whose: str  # 現在このセッション履歴が誰に利用されているかを表す
    purpose: str | None = None  # セッションの目的
    participant_profile: dict[str, str] | None = None  # 参加者のプロフィール

    def set_whose(self, whose: str) -> None:
        self.whose = whose
        self.clean_content()  # 所有者が変わった時点でその所有者にしか見えてはいけない内容を削除

    def set_purpose(self, purpose: str, participant_profile: dict[str, str]) -> None:  # 目的を変更する場合
        self.purpose = purpose
        self.participant_profile = participant_profile

    def clean_content(self) -> None:
        self.content = [item for item in self.content if isinstance(item, Statement)]

    def add_user_message(self, content: str) -> None:  # override
        self.content.append(
            Statement(role="user", content=content, whose=self.whose),
        )

    def add_system_message(self, content: str) -> None:  # override
        self.content.append(
            Statement(role="system", content=content, whose=self.whose),
        )

    def add_assistant_message(self, content: str) -> None:  # override
        self.content.append(
            Statement(role="assistant", content=content, whose=self.whose),
        )

    def get_content(self) -> list[dict]:  # override (現在の発言者に合わせて視点を変えて履歴を作成)
        ret_list = []
        ret_list.append(self._get_session_system_prompt())  # セッション用のシステムプロンプトを一番初めに追加
        for item in self.content:
            if isinstance(item, Statement):
                if item.whose == self.whose:  # 発言者が所有者と同じ場合
                    role = item.role
                    content_type = "input_text" if role == "system" else "output_text"
                    content = [
                        {"type": content_type, "text": item.content},
                    ]
                else:  # 発言者が所有者と異なる場合、立場を変換
                    if item.role == "system":  # 他のエージェントのシステムメッセージは無視
                        continue
                    role = "user"  # 所有者から見て他のエージェントはユーザに
                    content = [{"type": "input_text", "text": f"({item.whose}): " + item.content}]  # 発言元を明記
                ret_list.append({"role": role, "content": content})
            else:  # LLMプロバイダから返ってきたオブジェクトの場合はそのまま追加
                ret_list.append(item)
        return ret_list

    def _get_session_system_prompt(self) -> dict:
        text = (
            f"You are participating in a discussion session. Your name is {self.whose}. \n"
            f"The purpose of this session is:\n{self.purpose}\n\n The profiles of the participants are as follows: {self.participant_profile}.\n\n"
        )
        return {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                },
            ],
        }

    def is_finished(self) -> bool:
        self.clean_content()
        return len(self.content) >= 8  # noqa: PLR2004 一旦4回目以降のやり取りがある場合は終了とみなす
