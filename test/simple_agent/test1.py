from typing import Literal

from agent_template import Agent, BaseTool, OpenAILLM, tool


class MyTool(BaseTool):
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
        クラスに在籍している生徒の名前を取得するツール

        Returns:
            list[str]: 生徒の名前のリスト
        """
        return ["alice", "bob", "mike"]


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    tools = [MyTool()]
    agent = Agent(tools=tools, llm=llm, log_dir="./logs")

    system_prompt = (
        "あなたは優秀な学校の先生です。生徒の名前を聞かれたら、クラスに在籍している生徒の名前を教えてください。"
        "また、生徒の成績を聞かれたら、その生徒の成績を教えてください。"
    )
    task = "クラスに在籍している生徒の名前を教えてください。また、aliceの成績も教えてください。"

    final_response = agent.execute_task(system_prompt=system_prompt, task=task, use_log=False)
    print("Final Response:", final_response)
    total_fee = agent.get_total_fee()
    print(f"Total Fee: ${total_fee:.6f}")
