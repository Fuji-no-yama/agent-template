from abc import ABC, abstractmethod
from typing import Any

from agent_template._type.llm_responce import LLMResponse
from agent_template.history.history import History


class LLMInterface(ABC):
    @abstractmethod
    async def chat_with_history(
        self,
        history: History,
        *,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """LLMのAPIを使用してチャット履歴に基づく応答を生成します。

        Args:
            history (list[dict[str, str]]): チャット履歴のリスト。各辞書は"role"と"content"キーを含む。
            model (str | None, optional): 使用するモデル名.
            temperature (float, optional): 応答のランダム性を制御する値（0.0-2.0）. Defaults to 1.0.
            top_p (float, optional): nucleus samplingのパラメータ（0.0-1.0）. Defaults to 1.0.
            max_tokens (int, optional): 生成する最大トークン数. Defaults to 1024.

        Returns:
            LLMResponse: LLMからの応答を含むLLMResponseオブジェクト

        Raises:
            RetryableError: APIの一時的なエラー（レート制限、接続エラー）が発生した場合
        """

    @abstractmethod
    async def chat_with_history_tools(  # noqa: PLR0913
        self,
        history: History,
        *,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        tools: list[dict],
    ) -> LLMResponse:
        """LLMのAPIをツール機能付きでマルチターン履歴を使用して実行します。

        Args:
            history (list[dict[str, str]]): チャット履歴のリスト。各辞書は"role"と"content"キーを含む。
            model (str | None, optional): 使用するモデル名.
            temperature (float, optional): 応答のランダム性を制御する値（0.0-2.0）. Defaults to 1.0.
            top_p (float, optional): nucleus samplingのパラメータ（0.0-1.0）. Defaults to 1.0.
            max_tokens (int, optional): 生成する最大トークン数. Defaults to 1024.
            tools (list[dict]): 利用可能なツールの定義リスト(以下の形式)
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_calculation",
                            "description": "文字列で与えた数式を計算することができます。四則演算と()に対応しています。",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "expr": {"type": "string", "description": "計算する数式"},
                                },
                                "required": ["expr"],
                            },
                        },
                    },

        Returns:
            LLMResponse: LLMからの応答を含むLLMResponseオブジェクト

        Raises:
            RetryableError: APIの一時的なエラー（レート制限、接続エラー）が発生した場合
        """

    @abstractmethod
    def set_tool_result(self, history: History, result: dict[str, Any]) -> History:
        """履歴にツールの実行結果を追加します。

        Args:
            history (History): チャット履歴オブジェクト
            result (dict[str:str]): ツールの実行結果 (例: {"result_a": 1, "result_b": 2})

        Returns:
            History: ツールの実行結果が追加されたチャット履歴オブジェクト
        """
