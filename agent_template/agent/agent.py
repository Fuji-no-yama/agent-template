import asyncio
import json
import os
from logging import Logger
from pathlib import Path

from agent_template._interface.llm_interface import LLMInterface
from agent_template._other.config.settings import settings
from agent_template._other.util import get_logger
from agent_template._type.llm_responce import LLMResponse
from agent_template.history.history import History
from agent_template.tool.base_tool import BaseTool


class Agent:
    """
    エージェントを表すクラス
    """

    who_am_i: str
    llm: LLMInterface
    tools: list[BaseTool]
    log_dir: Path

    def __init__(self, who_am_i: str, tools: list[BaseTool], llm: LLMInterface, log_dir: os.PathLike) -> None:
        """
        エージェントオブジェクトを作成する。

        Args:
            who_am_i (str): このエージェントが何をするエージェントなのかを表す
            tools (list[BaseTool]): エージェントが使用するツールのリスト
            llm (LLMInterface): エージェントが使用するLLMインターフェース
            log_dir (os.PathLike): エージェントのログを保存するディレクトリパス
        """
        self.who_am_i = who_am_i
        self.llm = llm
        self.tools = tools
        self.log_dir = Path(log_dir)

    def execute_task(self, task: str, *, use_log: bool = False) -> str:
        """
        エージェントとしてツールを使いながらタスクを実行する

        Args:
            system_prompt (str): システムプロンプト
            task (str): ユーザからのタスク
            use_log (bool): ログを記録するかどうか

        Returns:
            str: タスクへの最終応答
        """
        history = History(content=[])
        if use_log:
            logger = get_logger(self.log_dir, file_prefix="execute_task")
            logger.info(f"[システムプロンプト]: {self.who_am_i}")
            logger.info(f"[ユーザタスク]: {task}")
        history.add_system_message(content=self.who_am_i)
        history.add_user_message(content=task)
        return self._execute_llm_loop(history, use_log=use_log, logger=logger if use_log else None)

    def execute_complex_task(self, task: str, *, use_log: bool = False) -> str:
        """
        エージェントとしてツールを使いながらタスクを実行する。(計画策定ステップ・実行ステップに分解)

        Args:
            system_prompt (str): システムプロンプト
            task (str): ユーザからのタスク

        Returns:
            str: タスクへの最終応答
        """

        go_to_next_step = False
        if use_log:
            logger = get_logger(self.log_dir, file_prefix="execute_complex_task")
            logger.info(f"[システムプロンプト]: {self.who_am_i}")
            logger.info(f"[ユーザタスク]: {task}")
        with (settings.data_dir / "prompt" / "complex_task_planning.prompt").open("r", encoding="utf-8") as f:
            planning_prompt = f.read()
        while not go_to_next_step:
            history = History(content=[])
            history.add_system_message(content=self.who_am_i)
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
                    logger.info(f"[計画ステップ最終応答]: {resp.content}") if use_log else None
                    break
                else:
                    break  # ツールが呼び出されてしまった場合は再度計画ステップを実行
        history.add_system_message(content="Please perform the tasks according to the plan above.")
        return self._execute_llm_loop(history, use_log=use_log, logger=logger if use_log else None)

    def _execute_llm_loop(self, history: History, *, use_log: bool = False, logger: Logger | None = None) -> str:
        while True:
            responses: list[LLMResponse] = asyncio.run(
                self.llm.chat_with_history_tools(
                    history=history,
                    tools=self.tools,
                ),
            )
            history = responses[-1].return_history  # 最終の履歴を取得
            for resp in responses:
                if not resp.is_tool_call:
                    logger.info(f"[実行ステップ最終応答]: {resp.content}") if use_log else None
                    return resp.content  # 最終応答を返す
                logger.info(f"[ツール指定]:\nname->{resp.tool_name}\nargs->{resp.tool_args}") if use_log else None
                try:
                    tool_res = self._execute_tool(resp)  # ツールを実行
                except Exception as e:  # noqa: BLE001
                    tool_res = f"Exception occurred during tool ({resp.tool_name}) execution: {e}"
                logger.info(f"[ツール結果]:\nresult->{tool_res}") if use_log else None
                history = self.llm.set_tool_result(
                    history=history,
                    tool_name=resp.tool_name,
                    tool_id=resp.tool_id,
                    result=tool_res,
                )

    def _execute_tool(self, llm_response: LLMResponse) -> str:  # LLMの出力に応じてツールを実行する内部関数
        tool_name = llm_response.tool_name  # これは関数名
        tool_args = llm_response.tool_args
        for tool_instance in self.tools:
            if tool_instance.has_tool(tool_name):
                tool_res = tool_instance.execute_tool(tool_name=tool_name, args=tool_args)
                if isinstance(tool_res, str):
                    return tool_res
                else:
                    return json.dumps(tool_res, ensure_ascii=False)

    def get_total_fee(self) -> float:
        """これまでのやり取りで発生した総費用を取得する。

        Returns:
            float: 総費用（ドル単位）
        """
        return self.llm.get_total_fee()
