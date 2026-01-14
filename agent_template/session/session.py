import json
from logging import Logger

from agent_template import Agent
from agent_template._other.util import get_logger
from agent_template._type.session_history import SessionHistory


class Session:
    """
    マルチエージェントによる話し合いを行う「セッション」を表すクラス
    """

    def __init__(self, participants: list[Agent]) -> None:
        self.participants: list[Agent] = participants

    def start_session(self, purpose: str, start_agent_name: str | None = None, *, use_log: bool = False) -> None:
        """
        セッションを開始する

        Args:
            purpose (str): このセッションで達成したい目的
            start_agent_name (str | None): セッションを開始するエージェントの名前。Noneの場合はランダムに選択。
        """
        participant_profile: dict[str, str] = {}  # 参加者のプロファイルを作成し登録
        for agent in self.participants:
            participant_profile[agent.name] = agent.who_am_i
        session_history = SessionHistory(content=[], whose="", purpose=purpose, participant_profile=participant_profile)

        if start_agent_name is None:
            current_agent: Agent = self.participants[0]
        else:
            for agent in self.participants:
                if agent.name == start_agent_name:
                    current_agent: Agent = agent
            err_msg = f"Agent with name {start_agent_name} not found among participants."
            raise ValueError(err_msg)

        logger: Logger = get_logger(log_dir="/workspace/tmp/log", file_prefix="session")

        while True:
            session_history.set_whose(current_agent.name)
            session_history: SessionHistory = current_agent._respond_to_history(
                history=session_history,
                use_log=use_log,
                logger=logger if use_log else None,
            )
            logger.info(f"{current_agent.name}から見た履歴")
            debug_history: list[dict] = session_history.get_content()
            cleaned_debug_history: list[dict] = []
            for item in debug_history:
                if isinstance(item, dict):
                    cleaned_debug_history.append(item)
                else:
                    cleaned_debug_history.append({"type": "object", "data": "おそらくツール呼び出し"})
            logger.info(json.dumps(cleaned_debug_history, ensure_ascii=False, indent=2))
            if all(agent._judge_finished(history=session_history) for agent in self.participants):  # エージェント全員が目的達成を認めた場合終了
                break
            current_agent: Agent = self._get_next_agent_from_score(history=session_history, logger=logger if use_log else None)  # 次の発言者を決定

    def _get_next_agent_from_score(self, history: SessionHistory, logger: Logger | None = None) -> Agent:  # 次の発言者を決定する
        max_score = -1
        next_agent: Agent = self.participants[0]
        for agent in self.participants:
            score: int | float = agent._get_motivation_score(history=history)
            if logger:
                logger.info(f"{agent.name}の動機化スコア: {score}")
            if score > max_score:
                max_score: int | float = score
                next_agent: Agent = agent
            elif score == max_score:
                if history.get_silence_count(name=agent.name) > history.get_silence_count(name=next_agent.name):  # 沈黙時間が長い方が優先
                    next_agent: Agent = agent
        if max_score == -1:
            err_msg = "All agents returned -1 motivation score; cannot determine next agent."
            raise ValueError(err_msg)
        return next_agent

    def get_total_fee(self) -> float:
        """
        セッション全体で使用されたコストを取得する(ドル単位)

        Returns:
            float: セッション全体で使用されたコスト(ドル単位)
        """
        total_fee = 0.0
        for agent in self.participants:
            total_fee += agent.llm.get_total_fee()
        return total_fee
