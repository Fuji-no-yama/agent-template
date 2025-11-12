from typing import Literal

from agent_template import Agent, BaseTool, OpenAILLM, Session, tool


class TeacherTool(BaseTool):
    @tool()
    def get_grade(self, name: Literal["alice", "bob", "mike"]) -> str:
        """
        生徒の名前に応じて成績を取得するツール

        Args:
            name (Literal["alice", "bob", "mike"]): 生徒の名前

        Returns:
            str: 成績の情報
        """
        grades = {
            "alice": "Aliceの成績は数学が90点、英語が85点、です。",
            "bob": "Bobの成績は数学が80点、英語が75点、です。",
            "mike": "Mikeの成績は数学が70点、英語が65点、です。",
        }
        return grades.get(name, "該当する生徒が見つかりません。")

    @tool()
    def get_name(self) -> list[str]:
        """
        クラスに在籍している生徒の名前と性別を取得するツール

        Returns:
            list[str]: 生徒の名前のリスト
        """
        return ["alice:female", "bob:male", "mike:male"]


class PrincipalTool(BaseTool):
    @tool()
    def decide_gradution(self, sex: Literal["male", "female"], average_score: float) -> bool:
        """
        生徒の性別と平均点に応じて卒業を決定するツール

        Args:
            sex (Literal["male", "female"]): 生徒の性別
            average_score (float): 生徒の平均点

        Returns:
            bool: 卒業可能かどうか
        """


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    tools = [TeacherTool()]
    teacher_agent = Agent(
        name="Teacher",
        who_am_i="あなたは優秀な学校の先生です。生徒の名前を聞かれたら、クラスに在籍している生徒の名前を教えてください。また、生徒の成績を聞かれたら、その生徒の成績を教えてください。",
        tools=tools,
        llm=llm,
        log_dir="./logs",
    )

    principal_agent = Agent(
        name="Principal",
        who_am_i="あなたは学校の校長です。生徒の性別と平均点に応じて卒業を決定します。",
        tools=[PrincipalTool()],
        llm=llm,
        log_dir="./logs",
    )

    session = Session(participants=[teacher_agent, principal_agent])
    session.start_session(
        purpose="在籍生徒について情報をもとに卒業の決定を判断する。",
        start_agent_name="Principal",
        use_log=True,
    )
    total_fee = session.get_total_fee()
    print(f"Total Fee: ${total_fee:.6f}")
