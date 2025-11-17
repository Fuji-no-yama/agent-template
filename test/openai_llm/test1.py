"""
ツール実行を行うテスト
"""

import asyncio

from agent_template import BaseTool, History, OpenAILLM, tool

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
