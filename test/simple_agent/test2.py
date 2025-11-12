from typing import Literal

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
    def create_study_plan(self, subjects: list[str], difficulty: Literal["easy", "normal", "hard"]) -> list[dict[str, str]]:
        """
        科目リストと難易度に基づいて学習プランを作成するツール

        Args:
            subjects (list[str]): 学習対象の科目リスト
            difficulty (Literal["easy", "normal", "hard"]): 学習の難易度

        Returns:
            list[dict[str, str]]: 各科目の学習プラン
        """
        time_mapping = {
            "easy": "30分",
            "normal": "60分",
            "hard": "90分",
        }

        method_mapping = {
            "easy": "基礎問題中心",
            "normal": "基礎+応用問題",
            "hard": "応用+発展問題",
        }

        study_time = time_mapping[difficulty]
        study_method = method_mapping[difficulty]

        # リスト内包表記を使用して計画を作成
        plans = [
            {
                "subject": subject,
                "duration": study_time,
                "method": study_method,
                "frequency": "週3回",
            }
            for subject in subjects
        ]

        return plans

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
        "あなたは教育データ分析の専門家です。学生のスコアデータを分析し、"
        "統計情報の計算、学習プランの作成、評価コメントの生成を行うことができます。"
        "与えられたデータを適切に処理し、分かりやすい形で結果を提示してください。"
    )

    # テスト1: list[int]型を使用するツールのテスト
    print("=== テスト1: スコア分析（list[int]型） ===")
    task1 = "次のテストスコアを分析してください: [85, 92, 78, 95, 88, 76, 89, 91, 83, 87]"

    try:
        response1 = agent.execute_task(system_prompt=system_prompt, task=task1, use_log=False)
        print("Response1:", response1)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test1: {e}")

    print("\n" + "=" * 60 + "\n")

    # テスト2: dict[str, int]型を使用するツールのテスト
    print("=== テスト2: 学生データ処理（dict[str, int]型） ===")
    task2 = """次の学生の成績データを処理し、評価コメントを生成してください:
    数学: 85, 英語: 92, 理科: 78, 社会: 88, 国語: 90"""

    # エージェントの料金をリセット
    agent.llm.input_token = 0
    agent.llm.output_token = 0

    try:
        response2 = agent.execute_task(system_prompt=system_prompt, task=task2, use_log=False)
        print("Response2:", response2)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test2: {e}")

    print("\n" + "=" * 60 + "\n")

    # テスト3: 複雑な型（list[str] + Literal）を使用するツールのテスト
    print("=== テスト3: 学習プラン作成（list[str] + Literal型） ===")
    task3 = "数学、英語、理科の3科目について、難易度「normal」で学習プランを作成してください。"

    # エージェントの料金をリセット
    agent.llm.input_token = 0
    agent.llm.output_token = 0

    try:
        response3 = agent.execute_task(system_prompt=system_prompt, task=task3, use_log=False)
        print("Response3:", response3)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test3: {e}")

    print("\n" + "=" * 60 + "\n")

    # テスト4: 最も複雑な型（list[dict[str, int]]）を使用するツールのテスト
    print("=== テスト4: クラス統計（list[dict[str, int]]型） ===")
    task4 = """クラス全体の成績データから統計を計算してください。以下は3人の学生データです:
    学生1: 数学85, 英語90, 理科88
    学生2: 数学92, 英語87, 理科95
    学生3: 数学78, 英語94, 理科82"""

    # エージェントの料金をリセット
    agent.llm.input_token = 0
    agent.llm.output_token = 0

    try:
        response4 = agent.execute_task(system_prompt=system_prompt, task=task4, use_log=False)
        print("Response4:", response4)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test4: {e}")

    print("\n🎉 全テスト完了!")
