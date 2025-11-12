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

from agent_template._interface import LLMInterface
from agent_template._other.config.settings import settings
from agent_template._other.exception import RetryableError
from agent_template._type import History, LLMResponse
from agent_template.tool import BaseTool, tool

if TYPE_CHECKING:
    from openai.types.responses import Response


@lru_cache
def _llm_client() -> AsyncOpenAI:  # キャッシュを使用するためモジュール関数としておく
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    return client


class OpenAILLM(LLMInterface):
    def __init__(self, model: str | None = "gpt-4.1", temperature: float | None = 0.0) -> None:
        if model not in settings.openai_model_price:
            err_msg = f"Unsupported model: {model}. Supported models are: {list(settings.openai_model_price.keys())}"
            raise ValueError(err_msg)
        self.model = model
        self.temperature = temperature
        self.output_token = 0
        self.input_token = 0

    def convert_type_info_to_schema(self, type_info: dict[str, Any]) -> dict[str, Any]:
        """
        BaseTool.get_tool_information()から得られる詳細な型情報を
        OpenAI Function Calling用のJSON Schemaに変換する。

        Args:
            type_info (dict): BaseTool._analyze_type_annotation()の出力

        Returns:
            dict: OpenAI API用のJSON Schema
        """
        schema = {"type": type_info["type"]}

        # Literal型（enum）の処理
        if "enum" in type_info:
            schema["enum"] = type_info["enum"]

        # Optional型（nullable）の処理
        if type_info.get("nullable", False):
            schema["nullable"] = True

        # array型の場合、items情報を追加
        if type_info["type"] == "array" and "items" in type_info:
            schema["items"] = self.convert_type_info_to_schema(type_info["items"])

        # object型の場合、additionalProperties情報を追加
        if type_info["type"] == "object" and "additionalProperties" in type_info:
            schema["additionalProperties"] = self.convert_type_info_to_schema(type_info["additionalProperties"])

        return schema

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
        top_p: float = 1.0,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "input": history.get_content(),
            "temperature": self.temperature,
            "top_p": top_p,
        }
        try:
            response: Response = await _llm_client().responses.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        self.input_token += response.usage.input_tokens
        self.output_token += response.usage.output_tokens
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
                    type_info = arg["type_info"]
                    property_schema = self.convert_type_info_to_schema(type_info)
                    property_schema["description"] = arg["description"]

                    arg_properties[arg["name"]] = property_schema

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
            "model": self.model,
            "input": history.get_content(),
            "temperature": self.temperature,
            "top_p": top_p,
            "tools": tool_for_param,
        }
        try:
            response: Response = await _llm_client().responses.create(**params)
        except (RateLimitError, APIConnectionError) as e:
            raise RetryableError(str(e)) from e
        self.input_token += response.usage.input_tokens
        self.output_token += response.usage.output_tokens
        ret_history = history
        ret_response: list[LLMResponse] = []

        for item in response.output:
            if item.type == "function_call":  # ツール呼び出しの場合
                ret_history.add_object(item)  # OpenAIの使用に合わせて帰ってきたオブジェクトをそのまま登録
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

    def set_tool_result(self, history: History, tool_name: str, tool_id: str, result: str) -> History:
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

    def get_total_fee(self) -> float:
        return (
            self.input_token
            * settings.openai_model_price.get(
                self.model,
                {"input": 0.0, "output": 0.0},
            )["input"]
            + self.output_token
            * settings.openai_model_price.get(
                self.model,
                {"input": 0.0, "output": 0.0},
            )["output"]
        )


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
