from typing import Literal

from agent_template import Agent, BaseTool, OpenAILLM, Session, tool


class DoctorTool(BaseTool):
    @tool()
    def get_medicine(self, disease_name: Literal["風邪", "インフルエンザ", "花粉症"]) -> str:
        """
        病気の名前を入力すると、その病気に対する処方薬を取得するツール

        Args:
            disease_name (Literal["風邪", "インフルエンザ", "花粉症"]): 病気の名前

        Returns:
            str: 処方薬
        """
        medicines = {
            "風邪": "解熱鎮痛薬",
            "インフルエンザ": "抗ウイルス薬",
            "花粉症": "抗ヒスタミン薬",
        }
        return medicines.get(disease_name, "該当する病気が見つかりません。")

    @tool()
    def diagnosis(self, symptoms: list[str]) -> list[str]:
        """
        症状から、可能性のある病気を取得するツール

        Args:
            symptoms (list[str]): 症状のリスト("発熱", "咳", "鼻水", "関節痛"の部分集合のみを受付)

        Returns:
            list[str]: 可能性のある病気のリスト
        """
        possible_diseases = {
            "発熱": ["風邪", "インフルエンザ"],
            "咳": ["花粉症", "インフルエンザ"],
            "鼻水": ["風邪", "花粉症"],
            "関節痛": ["インフルエンザ"],
        }
        return ["花粉症"]
        ret_list = []
        for symptom in symptoms:
            ret_list += possible_diseases.get(symptom, [])
        return list(set(ret_list))


class NurseTool(BaseTool):
    @tool()
    def search_pharmacy(self, medicine_name: Literal["解熱鎮痛薬", "抗ウイルス薬", "抗ヒスタミン薬"]) -> str:
        """
        処方された薬の名前から、行くべき薬局を取得するツール

        Args:
            medicine_name (Literal["解熱鎮痛薬", "抗ウイルス薬", "抗ヒスタミン薬"]): 薬の名前

        Returns:
            str: 薬局の名前
        """
        pahracies = {
            "解熱鎮痛薬": "ドラッグストアA",
            "抗ウイルス薬": "ドラッグストアB",
            "抗ヒスタミン薬": "ドラッグストアC",
        }
        return pahracies.get(medicine_name, "該当する薬局が見つかりません。")

    @tool()
    def get_root_to_pharmacy(self, pharmacy_name: Literal["ドラッグストアA", "ドラッグストアB", "ドラッグストアC"]) -> str:
        """
        薬局の名前を入力すると、その薬局へのルートを取得するツール

        Args:
            pharmacy_name (Literal["ドラッグストアA", "ドラッグストアB", "ドラッグストアC"]): 薬局の名前

        Returns:
            str: ルート
        """
        roots = {
            "ドラッグストアA": "医院を出て東に100m、左折して50m",
            "ドラッグストアB": "医院を出て西に150m、右折して30m",
            "ドラッグストアC": "医院を出て、上空に100km",
        }
        return roots.get(pharmacy_name, [])


class PatientTool(BaseTool):
    @tool()
    def get_symptoms(self) -> list[str]:
        """
        あなたの過去2日の記録を取得できるツール

        Returns:
            list[str]: 記録
        """
        return ["今日は鼻水がよく出る。熱は34.7度だ。", "今日は咳も出てきた。熱は相変わらず33.0度だ。"]


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4.1", temperature=0.0)
    doctor_agent = Agent(
        name="Doctor",
        who_am_i="あなたは優秀な医者です。病気の診断と処方を行うことができます。",
        tools=[DoctorTool()],
        llm=llm,
        log_dir="./logs",
    )

    nurse_agent = Agent(
        name="Nurse",
        who_am_i="あなたは優秀な看護師です。薬局の決定とルートの探索をすることができます。",
        tools=[NurseTool()],
        llm=llm,
        log_dir="./logs",
    )

    patient_agent = Agent(
        name="Patient",
        who_am_i="あなたの名前はBob、患者です。医者や看護師に相談し、処方された薬の情報を取得することができます。",
        tools=[PatientTool()],
        llm=llm,
        log_dir="./logs",
    )

    session = Session(participants=[doctor_agent, nurse_agent, patient_agent])
    session.start_session(
        purpose="患者についての診断・処方を行い、行くべき薬局への道のりを報告する。",
        start_agent_name="Patient",
        use_log=True,
    )
    total_fee = session.get_total_fee()
    print(f"Total Fee: ${total_fee:.6f}")
