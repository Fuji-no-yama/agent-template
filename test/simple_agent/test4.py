from agent_template import Agent, BaseTool, OpenAILLM, tool


class SimpleTool(BaseTool):
    @tool()
    def process_scores(self, scores: list[int]) -> str:
        """
        数値リストを処理して結果を返すツール

        Args:
            scores (list[int]): 処理対象の数値リスト

        Returns:
            str: 処理結果
        """
        total = sum(scores)
        average = total / len(scores) if scores else 0
        return f"合計: {total}, 平均: {average:.2f}, 件数: {len(scores)}"

    @tool()
    def process_dict(self, data: dict[str, int]) -> str:
        """
        辞書データを処理して結果を返すツール

        Args:
            data (dict[str, int]): 処理対象の辞書データ

        Returns:
            str: 処理結果
        """
        total = sum(data.values())
        items = list(data.items())
        return f"合計値: {total}, 項目数: {len(items)}, 内容: {dict(data)}"


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    tools = [SimpleTool()]
    agent = Agent(tools=tools, llm=llm, log_dir="./logs")

    system_prompt = (
        "あなたはデータ処理アシスタントです。"
        "データが与えられたら、必ず利用可能なツールを使用して処理してください。"
        "process_scoresツールは数値リストを処理し、process_dictツールは辞書データを処理します。"
    )

    print("=== 単純テスト1: リスト処理 ===")
    task1 = "process_scoresツールを使用して、この数値リスト [10, 20, 30, 40, 50] を処理してください。"

    try:
        agent.llm.input_token = 0
        agent.llm.output_token = 0

        response1 = agent.execute_task(system_prompt=system_prompt, task=task1, use_log=False)
        print("Response1:", response1)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test1: {e}")

    print("\n" + "=" * 50 + "\n")

    print("=== 単純テスト2: 辞書処理 ===")
    task2 = 'process_dictツールを使用して、この辞書データ {"math": 85, "english": 90, "science": 78} を処理してください。'

    try:
        agent.llm.input_token = 0
        agent.llm.output_token = 0

        response2 = agent.execute_task(system_prompt=system_prompt, task=task2, use_log=False)
        print("Response2:", response2)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test2: {e}")

    print("\n🎯 単純テスト完了!")
