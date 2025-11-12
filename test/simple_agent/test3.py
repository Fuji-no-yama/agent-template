from agent_template import Agent, BaseTool, OpenAILLM, tool


class DataProcessingTool(BaseTool):
    @tool()
    def analyze_scores(self, scores: list[int]) -> dict[str, float]:
        """
        スコアのリストを分析し、統計情報を返すツール

        Args:
            scores (list[int]): 分析対象のスコアのリスト

        Returns:
            dict[str, float]: 統計情報（平均、最大、最小）
        """
        if not scores:
            return {"average": 0.0, "max": 0.0, "min": 0.0}

        return {
            "average": sum(scores) / len(scores),
            "max": float(max(scores)),
            "min": float(min(scores)),
        }

    @tool()
    def process_student_data(self, student_info: dict[str, int]) -> str:
        """
        学生の情報を処理し、評価コメントを生成するツール

        Args:
            student_info (dict[str, int]): 学生の科目別スコア辞書

        Returns:
            str: 評価コメント
        """
        if not student_info:
            return "学生データがありません。"

        total_score = sum(student_info.values())
        subject_count = len(student_info)
        average = total_score / subject_count

        best_subject = max(student_info, key=student_info.get)
        worst_subject = min(student_info, key=student_info.get)

        return (
            f"総合評価：平均点 {average:.1f}点\n"
            f"最も得意な科目：{best_subject}（{student_info[best_subject]}点）\n"
            f"改善が必要な科目：{worst_subject}（{student_info[worst_subject]}点）"
        )

    @tool()
    def get_class_statistics(self, class_data: list[dict[str, int]]) -> dict[str, float]:
        """
        クラス全体のデータから統計情報を計算するツール

        Args:
            class_data (list[dict[str, int]]): クラス全体の学生データ（各学生の科目別スコア）

        Returns:
            dict[str, float]: クラス全体の統計情報
        """
        if not class_data:
            return {"class_average": 0.0, "total_students": 0.0}

        all_scores = []
        total_students = len(class_data)

        for student_data in class_data:
            all_scores.extend(student_data.values())

        class_average = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "class_average": class_average,
            "total_students": float(total_students),
            "total_subjects_count": float(len(all_scores)),
        }


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    tools = [DataProcessingTool()]
    agent = Agent(tools=tools, llm=llm, log_dir="./logs")

    system_prompt = (
        "あなたは教育データ分析の専門家です。"
        "与えられたデータをprocess_student_dataやget_class_statisticsなどの"
        "利用可能なツールを使って処理し、分析結果を提供してください。"
        "データが与えられた場合は、必ずツールを使用して処理してください。"
    )

    # より明示的なテスト2: dict[str, int]型
    print("=== 明示的テスト: 学生データ処理（dict[str, int]型） ===")
    task2_explicit = (
        "以下の学生の成績データを process_student_data ツールを使って分析してください。\n"
        "学生の成績データ（辞書形式）:\n"
        '{"数学": 85, "英語": 92, "理科": 78, "社会": 88, "国語": 90}'
    )

    try:
        # エージェントの料金をリセット
        agent.llm.input_token = 0
        agent.llm.output_token = 0

        response2 = agent.execute_task(system_prompt=system_prompt, task=task2_explicit, use_log=False)
        print("Response2:", response2)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in explicit test2: {e}")

    print("\n" + "=" * 60 + "\n")

    # より明示的なテスト4: list[dict[str, int]]型
    print("=== 明示的テスト: クラス統計（list[dict[str, int]]型） ===")
    task4_explicit = (
        "以下のクラス全体のデータを get_class_statistics ツールを使って統計を計算してください。\n"
        "クラスデータ（リスト形式）:\n"
        "[\n"
        '  {"数学": 85, "英語": 90, "理科": 88},\n'
        '  {"数学": 92, "英語": 87, "理科": 95},\n'
        '  {"数学": 78, "英語": 94, "理科": 82}\n'
        "]"
    )

    try:
        # エージェントの料金をリセット
        agent.llm.input_token = 0
        agent.llm.output_token = 0

        response4 = agent.execute_task(system_prompt=system_prompt, task=task4_explicit, use_log=False)
        print("Response4:", response4)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in explicit test4: {e}")

    print("\n🎉 明示的テスト完了!")

    # ツールスキーマの確認
    print("\n=== 生成されたツールスキーマの確認 ===")
    tool_instance = DataProcessingTool()
    tool_info_list = tool_instance.get_tool_information()

    for tool_info in tool_info_list:
        print(f"\n📋 ツール名: {tool_info['name']}")
        for arg in tool_info["args"]:
            print(f"  引数 {arg['name']}: {arg['type_info']}")
            # OpenAI API形式への変換結果も表示
            converted_schema = llm.convert_type_info_to_schema(arg["type_info"])
            print(f"  → OpenAI Schema: {converted_schema}")
        print("-" * 40)
