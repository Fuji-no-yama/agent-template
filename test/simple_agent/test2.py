from pathlib import Path
from typing import Literal

from agent_template import Agent, BaseTool, OpenAILLM, tool


class DataProcessingTool(BaseTool):
    @tool()
    def analyze_scores(self, scores: list[int]) -> list[str]:
        """
        Tool to analyze a list of scores and perform 5-level evaluation (A, B, C, D, E) based on the scores

        Args:
            scores (list[int]): List of scores to analyze

        Returns:
            list[str]: List of 5-level evaluations corresponding to each score
        """
        ret_list: list[str] = []
        for score in scores:
            if score >= 90:  # noqa: PLR2004
                ret_list.append("A")
            elif score >= 80:  # noqa: PLR2004
                ret_list.append("B")
            elif score >= 70:  # noqa: PLR2004
                ret_list.append("C")
            elif score >= 60:  # noqa: PLR2004
                ret_list.append("D")
            else:
                ret_list.append("E")
        return ret_list

    @tool()
    def generate_eval_comments(self, subjects: list[str], score_level: list[str]) -> list[str]:
        """
        Tool to generate evaluation comments from 5-level evaluations for each subject

        Args:
            subjects (list[str]): List of subject names. Example: ["Math", "English", "Science"]
            score_level (list[str]): List of 5-level evaluations corresponding to each subject. Example: ["A", "B", "C"]

        Returns:
            list[str]: List of evaluation comments
        """
        comments: list[str] = []
        for subject, score in zip(subjects, score_level, strict=False):
            if not isinstance(subject, str):
                msg = f"Subject name must be a string. Got: {type(subject)}"
                raise TypeError(msg)
            if not isinstance(score, str):
                msg = f"Score for {subject} must be a string. Got: {type(score)}"
                raise TypeError(msg)
            if score not in ["A", "B", "C", "D", "E"]:
                msg = f"Score for {subject} must be one of ['A', 'B', 'C', 'D', 'E']. Got: {score}"
                raise ValueError(msg)

            if score == "A":  # noqa: PLR2004
                comments.append(f"Excellent performance in {subject}! Keep up the great work.")
            elif score == "B":  # noqa: PLR2004
                comments.append(f"Good performance in {subject}. Almost perfect!")
            elif score == "C":  # noqa: PLR2004
                comments.append(f"Average performance in {subject}. With effort, you can improve.")
            elif score == "D":  # noqa: PLR2004
                comments.append(f"You need to work harder in {subject}. Review is recommended.")
            else:  # score == "E"
                comments.append(f"Poor performance in {subject}. Seek active support.")
        return comments

    @tool()
    def get_student_scores(self) -> dict[str, dict[str, int]]:
        """
        Tool to return student names and their scores for each subject

        Returns:
            dict[str, dict[str, int]]: Dictionary with student names as keys and score dictionaries for each subject as values. Example: {"Taro Yamada": {"Math": 85, "English": 90, ...}, ...}
        """
        student_scores = {
            "Taro Yamada": {"Math": 85, "English": 90, "Science": 88},
            "Hanako Tanaka": {"Math": 92, "English": 87, "Science": 95},
            "Ichiro Suzuki": {"Math": 78, "English": 94, "Science": 82},
        }
        return student_scores


if __name__ == "__main__":
    my_tool = DataProcessingTool()
    import json

    for r in my_tool.get_tool_information():
        print(json.dumps(r, ensure_ascii=False, indent=2))

    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    tools = [DataProcessingTool()]
    agent = Agent(
        name="DataProcessor",
        who_am_i="You are professor. You analyze student scores and generate evaluation comments based on their performance.",
        tools=tools,
        llm=llm,
        log_dir=Path("/workspace/tmp/log"),
    )

    print("=== テスト: 評価コメント生成ツール ===")
    task = "Student Tanaka has a score of A in Math, B in Japanese, and C in Science. Please generate evaluation comments based on his scores."
    try:
        response = agent.execute_task(task=task, use_log=True)
        print("Response:", response)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test1: {e}")

    print("=== テスト: 全体テスト ===")
    task = "Please create report cards with comments on grades for all students."
    try:
        response = agent.execute_task(task=task, use_log=True)
        print("Response:", response)
        print(f"Fee: ${agent.get_total_fee():.6f}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error in test1: {e}")
