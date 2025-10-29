import asyncio
import json
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

    def __init__(self, tools: list[BaseTool], llm: LLMInterface) -> None:
        self.llm = llm
        self.tools = tools

    def execute_task(self, system_prompt: str, task: str) -> str:
        """
        エージェントとしてツールを使いながらタスクを実行する

        Args:
            system_prompt (str): システムプロンプト
            task (str): ユーザからのタスク

        Returns:
            str: タスクへの最終応答
        """
        history = History(content=[])
        history.add_system_message(content=system_prompt)
        history.add_user_message(content=task)
        while True:
            responses: list[LLMResponse] = asyncio.run(
                self.llm.chat_with_history_tools(
                    history=history,
                    tools=self.tools,
                ),
            )
            current_history = responses[-1].return_history  # 最終の履歴を取得
            for resp in responses:
                if not resp.is_tool_call:
                    return resp.content  # 最終応答を返す
                tool_res = self._execute_tool(resp)  # ツールを実行
                current_history = self.llm.set_tool_result(
                    history=current_history,
                    tool_name=resp.tool_name,
                    tool_id=resp.tool_id,
                    result=tool_res,
                )

    def execute_complex_task(self, system_prompt: str, task: str) -> str:
        """
        エージェントとしてツールを使いながらタスクを実行する。(計画策定ステップ・実行ステップに分解)

        Args:
            system_prompt (str): システムプロンプト
            task (str): ユーザからのタスク

        Returns:
            str: タスクへの最終応答
        """
        go_to_next_step = False
        with (settings.data_dir / "prompt" / "complex_task_planning.prompt").open("r", encoding="utf-8") as f:
            planning_prompt = f.read()
        while not go_to_next_step:
            history = History(content=[])
            history.add_system_message(content=system_prompt)
            history.add_user_message(content=task)
            history.add_system_message(content=planning_prompt)
            responses: list[LLMResponse] = asyncio.run(
                self.llm.chat_with_history_tools(
                    history=history,
                    tools=self.tools,
                ),
            )
            history = responses[-1].return_history  # 最終の履歴を取得
            for resp in responses:
                if not resp.is_tool_call:
                    go_to_next_step = True
                    print("計画ステップ:\n", resp.content)
                    print("==============================")
                    break
                else:
                    break  # ツールが呼び出されてしまった場合は再度計画ステップを実行

        history.add_system_message(content="Please perform the tasks according to the plan above.")
        while True:
            responses: list[LLMResponse] = asyncio.run(
                self.llm.chat_with_history_tools(
                    history=history,
                    tools=self.tools,
                ),
            )
            current_history = responses[-1].return_history  # 最終の履歴を取得
            for resp in responses:
                if not resp.is_tool_call:
                    return resp.content  # 最終応答を返す
                tool_res = self._execute_tool(resp)  # ツールを実行
                current_history = self.llm.set_tool_result(
                    history=current_history,
                    tool_name=resp.tool_name,
                    tool_id=resp.tool_id,
                    result=tool_res,
                )

    def _execute_tool(self, llm_response: LLMResponse) -> dict[str, Any]:  # LLMの出力に応じてツールを実行する内部関数
        tool_name = llm_response.tool_name  # これは関数名
        tool_args = llm_response.tool_args
        for tool_instance in self.tools:
            if tool_instance.has_tool(tool_name):
                return json.dumps(tool_instance.execute_tool(tool_name=tool_name, args=tool_args))
