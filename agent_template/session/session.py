import json
import random

from agent_template import Agent
from agent_template._other.util import get_logger
from agent_template._type.session_history import SessionHistory


class Session:
    """
    マルチエージェントによる話し合いを行う「セッション」を表すクラス
    """

    def __init__(self, participants: list[Agent]) -> None:
        self.participants = participants

    def start_session(self, purpose: str, start_agent_name: str | None = None, *, use_log: bool = False) -> None:
        """
        セッションを開始する

        Args:
            purpose (str): このセッションで達成したい目的
            start_agent_name (str | None): セッションを開始するエージェントの名前。Noneの場合はランダムに選択。
        """
        participant_profile = {}  # 参加者のプロファイルを作成し登録
        for agent in self.participants:
            participant_profile[agent.name] = agent.who_am_i
        session_history = SessionHistory(content=[], whose="", purpose=purpose, participant_profile=participant_profile)

        if start_agent_name is None:
            current_agent = random.choice(self.participants)
        else:
            current_agent = next(agent for agent in self.participants if agent.name == start_agent_name)

        logger = get_logger(log_dir="/workspace/tmp/log", file_prefix="session")

        while True:
            session_history.set_whose(current_agent.name)
            session_history = current_agent._respond_to_history(history=session_history, use_log=use_log, logger=logger if use_log else None)
            logger.info(f"{current_agent.name}から見た履歴")
            debug_history = session_history.get_content()
            cleaned_debug_history = []
            for item in debug_history:
                if isinstance(item, dict):
                    cleaned_debug_history.append(item)
                else:
                    cleaned_debug_history.append({"type": "object", "data": "おそらくツール呼び出し"})
            logger.info(json.dumps(cleaned_debug_history, ensure_ascii=False, indent=2))
            if session_history.is_finished():  # 終了の場合
                break
            current_agent = self._get_next_agent(history=session_history)  # 次の発言者を決定

    def _get_next_agent(self, history: SessionHistory) -> Agent:  # 次の発言者を決定する(前の発言者以外からランダム)
        candidates = [agent for agent in self.participants if agent.name != history.whose]
        next_agent = random.choice(candidates)
        return next_agent

    def _get_next_agent_from_score(self, history: SessionHistory) -> Agent:  # 次の発言者を決定する
        max_score = -1
        next_agent = None
        for agent in self.participants:
            score = agent._get_motivation_score(history=history)
            if score > max_score:
                max_score = score
                next_agent = agent
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
