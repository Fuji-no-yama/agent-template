from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    RateLimitError,
)
from settings import settings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from agent_template._other.exception import RetryableError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage


@lru_cache
def _llm_client() -> AsyncOpenAI:  # キャッシュを使用するためモジュール関数としておく
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    return client


class LLM:
    def __init__(self) -> None:
        pass

    @retry(
        retry=retry_if_exception_type(RetryableError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1.0),
        reraise=True,
    )
    async def chat_with_history(
        self,
        history: list[dict[str, str]],
        *,
        model: str | None = "gpt-4.1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        """OpenAI Chat Completions APIを使用してチャット履歴に基づく応答を生成します。

        Args:
            history (list[dict[str, str]]): チャット履歴のリスト。各辞書は"role"と"content"キーを含む。
            model (str | None, optional): 使用するOpenAIモデル名. Defaults to "gpt-4.1".
            temperature (float, optional): 応答のランダム性を制御する値（0.0-2.0）. Defaults to 1.0.
            top_p (float, optional): nucleus samplingのパラメータ（0.0-1.0）. Defaults to 1.0.
            max_tokens (int, optional): 生成する最大トークン数. Defaults to 1024.

        Returns:
            str: LLMからの応答テキスト

        Raises:
            RetryableError: APIの一時的なエラー（レート制限、接続エラー）が発生した場合
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": history,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response: ChatCompletion = await _llm_client().chat.completions.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        return response.choices[0].message.content

    @retry(
        retry=retry_if_exception_type(RetryableError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1.0),
        reraise=True,
    )
    async def chat_with_history_tools(  # noqa: PLR0913
        self,
        history: list[dict[str, str]],
        *,
        model: str | None = "gpt-4.1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        tools: list[dict],
    ) -> ChatCompletionMessage:
        """OpenAI Chat Completions APIをツール機能付きでマルチターン履歴を使用して実行します。

        Args:
            history (list[dict[str, str]]): チャット履歴のリスト。各辞書は"role"と"content"キーを含む。
            model (str | None, optional): 使用するOpenAIモデル名. Defaults to "gpt-4.1".
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
            ChatCompletionMessage: LLMからの完全な応答メッセージオブジェクト(toolを使う場合は以下の形式)
                ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    annotations=[],
                    audio=None,
                    function_call=None,
                    tool_calls=[ChatCompletionMessageFunctionToolCall(
                        id='call_Yq0fBWrarWb8E5YgUVkSvgi7',
                        function=Function(arguments='{"expr":"1450*0.82*18763/156"}',
                        name='execute_calculation'),
                        type='function')
                    ]
                )

        Raises:
            RetryableError: APIの一時的なエラー（レート制限、接続エラー）が発生した場合
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": history,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "tools": tools,
        }
        try:
            response: ChatCompletion = await _llm_client().chat.completions.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        return response.choices[0].message


if __name__ == "__main__":
    llm = LLM()
    tools = [
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
    ]

    prompt = "What is the capital of France?"
    response = asyncio.run(llm.chat_with_history([{"role": "user", "content": prompt}]))
    print(response)

    prompt = "18763人で入場料が一人当たり1450円のテーマパークに入場し、18%の割引券を使った場合の合計金額は何ドルですか? 1ドルは156円です。"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=[{"role": "user", "content": prompt}],
            tools=tools,
        ),
    )
    print(response)

    prompt = "フランスの首都は?"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=[{"role": "user", "content": prompt}],
            tools=tools,
        ),
    )
    print(response)
