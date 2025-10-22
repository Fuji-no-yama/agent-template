import asyncio
from typing import TYPE_CHECKING

from agent_template._interface.llm_interface import LLMInterface
from agent_template._other.config.settings import settings
from agent_template._type.history import History
from agent_template.tool.base_tool import BaseTool

if TYPE_CHECKING:
    from agent_template._type.llm_responce import LLMResponse


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
        """エージェントとしてタスクを実行する関数"""
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
            tool_res = {}
            history = self.llm.set_tool_result(res.return_history, tool_res)
