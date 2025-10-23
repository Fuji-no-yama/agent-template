from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from agent_template._interface.llm_interface import LLMInterface
from agent_template._other.config.settings import settings
from agent_template._other.exception.exception import RetryableError
from agent_template._type.llm_responce import LLMResponse
from agent_template.history.history import History

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall


@lru_cache
def _llm_client() -> AsyncOpenAI:  # キャッシュを使用するためモジュール関数としておく
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    return client


class OpenAILLM(LLMInterface):
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
        history: History,
        *,
        model: str | None = "gpt-4.1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": model,
            "messages": history.content,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response: ChatCompletion = await _llm_client().chat.completions.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        ret_history = history
        ret_history.add(role="assistant", content=response.choices[0].message.content)
        return LLMResponse(
            content=response.choices[0].message.content,
            is_tool_call=False,
            tool_name=None,
            tool_args=None,
            return_history=ret_history,
        )

    @retry(
        retry=retry_if_exception_type(RetryableError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1.0),
        reraise=True,
    )
    async def chat_with_history_tools(  # noqa: PLR0913
        self,
        history: History,
        *,
        model: str | None = "gpt-4.1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        tools: list[dict],
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": model,
            "messages": history.content,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "tools": tools,
        }
        try:
            response: ChatCompletion = await _llm_client().chat.completions.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        ret_history = history
        ret_history.add(role="assistant", content=response.choices[0].message.content)
        if response.choices[0].message.tool_calls is not None:  # ツール呼び出しがある場合
            tool_call: ChatCompletionMessageToolCall = response.choices[0].message.tool_calls[0]
            ret_history.content.append(  # OpenAIの仕様に合わせて履歴にツール登録
                {
                    "role": "assistant",
                    "tool_call": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        },
                    ],
                },
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                is_tool_call=True,
                tool_name=tool_call.function.name,
                tool_args=tool_call.function.arguments,
                return_history=ret_history,
            )
        else:  # ツール呼び出しがない場合
            return LLMResponse(
                content=response.choices[0].message.content,
                is_tool_call=False,
                tool_name=None,
                tool_args=None,
                return_history=ret_history,
            )

    def set_tool_result(self, history: History, result: dict[str, Any]) -> History:
        if history.content[-1]["role"] != "tool":
            err_msg = f"The last message role is {history.content[-1]['role']}, expected 'tool'."
            raise ValueError(err_msg)
        history.content.append(
            {
                "role": "tool",
                "tool_call_id": history.content[-1]["tool_call"][0]["id"],
                "content": f"{result}",
            },
        )
        return history


if __name__ == "__main__":
    llm = OpenAILLM()
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
    response = asyncio.run(llm.chat_with_history(History(content=[{"role": "user", "content": prompt}])))
    print(response.content)

    prompt = "18763人で入場料が一人当たり1450円のテーマパークに入場し、18%の割引券を使った場合の合計金額は何ドルですか? 1ドルは156円です。"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=History(content=[{"role": "user", "content": prompt}]),
            tools=tools,
        ),
    )
    print(response.content)
    print(response.tool_name)
    print(response.tool_args)

    prompt = "フランスの首都は?"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=History(content=[{"role": "user", "content": prompt}]),
            tools=tools,
        ),
    )
    print(response.content)
    print(response.tool_name)
    print(response.tool_args)
