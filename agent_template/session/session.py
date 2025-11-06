from agent_template import Agent


class Session:
    """
    マルチエージェントによる話し合いを行う「セッション」を表すクラス
    """

    def __init__(self, participants: list[Agent]) -> None:
        self.participants = participants

    def start_session(self, purpose: str) -> None:
        """
        セッションを開始する

        Args:
            purpose (str): このセッションで達成したい目的
        """
