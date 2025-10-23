import asyncio
from typing import Any

from agent_template._interface.llm_interface import LLMInterface
from agent_template._other.config.settings import settings
from agent_template._type.llm_responce import LLMResponse
from agent_template.history.history import History
from agent_template.tool.base_tool import BaseTool


class Agent:
    """メモ
    エージェントの基底クラス
    """

    llm: LLMInterface
    system_prompt: str

    def __init__(self, tools: list[BaseTool], llm_provider: str = "openai") -> None:
        self.llm = settings.llms[llm_provider]()
        self.tools = tools

    def execute_task(self, system_prompt: str) -> None:
        """
        エージェントとしてタスクを実行するメソッド
        """
        history = History(content=[])
        history.add(role="system", content=system_prompt)
        while True:
            res: LLMResponse = asyncio.run(
                self.llm.chat_with_history_tools(
                    history=history,
                    tools=[tool.to_tool_definition() for tool in self.tools],
                ),
            )
            if not res.is_tool_call:
                break
            # ここでツール実行
            tool_res = self._execute_tool(res)
            history = self.llm.set_tool_result(res.return_history, tool_res)  # ツール実行結果を登録

    def _execute_tool(self, llm_response: LLMResponse) -> dict[str, Any]:  # LLMの出力に応じてツールを実行する内部関数
        tool_name = llm_response.tool_name
        tool_args = llm_response.tool_args
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(tool_args)
        return {}
