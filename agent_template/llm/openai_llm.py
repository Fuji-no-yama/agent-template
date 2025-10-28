from __future__ import annotations

import asyncio
import json
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
from agent_template.tool.base_tool import BaseTool, tool

if TYPE_CHECKING:
    from openai.types.responses import Response


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
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": model,
            "input": history.content,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            response: Response = await _llm_client().responses.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        ret_history = history
        ret_history.add_assistant_message(content=response.output_text)
        return LLMResponse(
            content=response.output_text,
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
        tools: list[BaseTool],
    ) -> list[LLMResponse]:
        tool_for_param = []  # ここでOpenAI API形式にツールを変換
        for tool_instance in tools:
            tool_info_list = tool_instance.get_tool_information()
            for tool_info in tool_info_list:
                arg_properties = {}
                required_args = []
                for arg in tool_info["args"]:
                    arg_properties[arg["name"]] = {
                        "type": arg["type"],
                        "description": arg["description"],
                    }
                    if arg.get("required", False):
                        required_args.append(arg["name"])
                tool_for_param.append(
                    {
                        "type": "function",
                        "name": tool_info["name"],
                        "description": tool_info["description"],
                        "parameters": {
                            "type": "object",
                            "properties": arg_properties,
                            "required": required_args,
                            "additionalProperties": False,
                        },
                    },
                )
        params: dict[str, Any] = {
            "model": model,
            "input": history.content,
            "temperature": temperature,
            "top_p": top_p,
            "tools": tool_for_param,
        }
        try:
            response: Response = await _llm_client().responses.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        ret_history = history
        ret_response: list[LLMResponse] = []

        for item in response.output:
            if item.type == "function_call":  # ツール呼び出しの場合
                ret_history.content.append(item)  # OpenAIの使用に合わせて帰ってきたオブジェクトをそのまま登録
                ret_response.append(
                    LLMResponse(
                        content=response.output_text,
                        is_tool_call=True,
                        tool_name=item.name,
                        tool_id=item.call_id,
                        tool_args=json.loads(item.arguments),
                        return_history=ret_history,
                    ),
                )
            else:  # 通常の応答の場合
                ret_history.add_assistant_message(content=response.output_text)
                ret_response.append(
                    LLMResponse(
                        content=response.output_text,
                        is_tool_call=False,
                        tool_name=None,
                        tool_id=None,
                        tool_args=None,
                        return_history=ret_history,
                    ),
                )
        return ret_response

    def set_tool_result(self, history: History, tool_name: str, tool_id: str, result: dict[str, Any]) -> History:
        history.content.append(
            {
                "type": "function_call_output",
                "call_id": tool_id,
                "output": str(
                    {
                        tool_name: result,
                    },
                ),
            },
        )
        return history


if __name__ == "__main__":
    llm = OpenAILLM()

    class MyTool(BaseTool):
        @tool(use_docstring=True)
        def add(self, x: int, y: int) -> int:
            """
            2つの整数を加算します。

            Args:
                x (int): 加算する最初の整数
                y (int): 加算する2番目の整数

            Returns:
                int: 加算結果
            """
            return x + y

        @tool(use_docstring=True)
        def multiply(self, a: float, b: float) -> float:
            """
            2つの浮動小数点数を乗算します。

            Args:
                a (float): 乗算する最初の浮動小数点数
                b (float): 乗算する2番目の浮動小数点数

            Returns:
                float: 乗算結果
            """
            return a * b

    prompt = "What is the capital of France?"
    response = asyncio.run(llm.chat_with_history(History(content=[{"role": "user", "content": prompt}])))
    print(response.content)

    prompt = "17345.987 * 34827.9 = ?"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=History().add_user_message(content=prompt),
            tools=[MyTool()],
        ),
    )
    for resp in response:
        print(resp.content)
        print(resp.tool_name)
        print(resp.tool_args)

    prompt = "フランスの首都は?"
    response = asyncio.run(
        llm.chat_with_history_tools(
            history=History().add_user_message(content=prompt),
            tools=[MyTool()],
        ),
    )
    for resp in response:
        print(resp.content)
        print(resp.tool_name)
        print(resp.tool_args)
