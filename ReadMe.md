# Agent Template

OpenAI API ネイティブなagentを作るためのライブラリです。カスタムツールを持つAIエージェントの作成と、マルチエージェントセッションでの協調作業を実現できます。

## 特徴

- 🤖 **シンプルなエージェント作成**: OpenAI APIを使用してカスタムツールを持つエージェントを簡単に作成
- 🛠️ **柔軟なツールシステム**: デコレータベースでカスタムツールを定義
- 👥 **マルチエージェント対応**: 複数のエージェントが協調して問題解決を行うセッション機能
- 📊 **コスト管理**: API使用料金の追跡機能

特にOpenAI Agent SDKと比較すると、**人間の対話を再現したマルチエージェントセッション**と**クラスとしてツールを構築できる点**が大きな差分となります。

## インストール

### 特定のバージョンを指定
mainブランチの更新等に伴い、機能が一部変更となる場合があるため、バージョンを固定してインストール・更新を行うことをお勧めします。

#### pipの場合  
- インストール
    ```bash
    pip install git+https://github.com/Fuji-no-yama/agent-template.git@1.0.0
    ```
- アップデート
    ```bash
    pip install -U git+https://github.com/Fuji-no-yama/agent-template.git@1.0.0
    ```

#### uvの場合
- インストール・アップデート
    ```bash
    uv add git+https://github.com/Fuji-no-yama/agent-template --tag 1.0.0
    ```


### GitHub mainブランチから直接インストール

```bash
pip install git+https://github.com/Fuji-no-yama/agent-template.git
```

## 必要な環境変数

OpenAI APIキーを環境変数に設定してください：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 基本的な使用方法

### 1. カスタムツールの作成

```python
from typing import Literal
from agent_template import BaseTool, tool

class MyTool(BaseTool):
    @tool(use_docstring=True)
    def get_weather(self, city: str) -> str:
        """
        指定された都市の天気情報を取得します。
        
        Args:
            city (str): 都市名
            
        Returns:
            str: 天気情報
        """
        # 実際の天気API呼び出しをここに実装
        return f"{city}の天気は晴れです"
    
    @tool(use_docstring=True)
    def calculate(self, operation: Literal["add", "subtract", "multiply"], x: int, y: int) -> int:
        """
        基本的な計算を行います。
        
        Args:
            operation (Literal["add", "subtract", "multiply"]): 計算の種類
            x (int): 最初の数値
            y (int): 2番目の数値
            
        Returns:
            int: 計算結果
        """
        if operation == "add":
            return x + y
        elif operation == "subtract":
            return x - y
        elif operation == "multiply":
            return x * y
```

### 2. シンプルなエージェントの作成と実行

```python
from agent_template import Agent, OpenAILLM

# LLMインスタンスの作成
llm = OpenAILLM(model="gpt-4.1", temperature=0.0)

# エージェントの作成
agent = Agent(
    name="WeatherAssistant",
    who_am_i="あなたは天気情報と計算を提供するアシスタントです。",
    tools=[MyTool()],
    llm=llm,
    log_dir="./logs"
)

# タスクの実行
response = agent.execute_task("東京の天気を教えて、そして5と3を掛け算してください")
print(response)

# 使用料金の確認
print(f"使用料金: ${agent.get_total_fee():.6f}")
```

### 3. 複雑なタスクの実行

```python
# 計画→実行の2段階処理で複雑なタスクを処理
response = agent.execute_complex_task(
    "今日の東京の天気を調べて、もし晴れなら10×5を、雨なら20÷4を計算してください"
)
print(response)
```

### 4. マルチエージェントセッション

```python
from agent_template import Session

# 複数のエージェントを作成
teacher_agent = Agent(
    name="Teacher",
    who_am_i="あなたは学校の先生です。生徒の成績管理を行います。",
    tools=[TeacherTool()],
    llm=llm,
    log_dir="./logs"
)

principal_agent = Agent(
    name="Principal", 
    who_am_i="あなたは学校の校長です。最終的な判断を行います。",
    tools=[PrincipalTool()],
    llm=llm,
    log_dir="./logs"
)

# セッションの開始
session = Session(participants=[teacher_agent, principal_agent])
session.start_session(
    purpose="生徒の卒業判定を行う",
    start_agent_name="Principal",
    use_log=True
)

print(f"セッション合計費用: ${session.get_total_fee():.6f}")
```

## ツールの詳細説明

ツールの引数として以下の型をサポートしています：

- **基本型**: `str`, `int`, `float`, `bool`
- **コレクション**: `list[T]`, `dict[str, T]`
- **Optional**: `Optional[T]` または `T | None`
- **Literal**: `Literal["option1", "option2", ...]`
- **複合型**: `list[dict[str, int]]` など

**ただし、LLMがこの形の引数を生成する必要があるため、引数の型が複雑になる程、失敗する率は高くなります。**
例えば、`list[dict[str, int]]`を引数に渡すよりかは、`list[str]`と`list[int]`の2つの引数に分割して渡した方が動作が安定します。

```python
@tool(use_docstring=True)
def complex_function(
    items: list[str],
    config: dict[str, int],
    mode: Literal["fast", "normal", "slow"],
    count: int | None = None
) -> dict[str, any]:
    """複雑な型を使用した関数の例"""
    # 実装
```

## 主要クラス

### Agent
個別のAIエージェントを表現するクラス

**主要メソッド**:
- `execute_task(task: str)`: シンプルなタスク実行
- `execute_complex_task(task: str)`: 計画→実行の2段階処理
- `get_total_fee()`: 使用料金の取得

### BaseTool
カスタムツールを作成するためのベースクラス

**主要デコレータ**:
- `@tool(use_docstring=True)`: メソッドをツールとして登録

### OpenAILLM
OpenAI APIとのインターフェースを提供

**対応モデル**:
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Session
マルチエージェントの協調セッションを管理

## ログ機能

`use_log=True` を設定することで、詳細なログを出力できます：

```python
agent.execute_task("タスクの内容", use_log=True)
```

ログには以下の情報が含まれます：
- システムプロンプト
- ユーザータスク
- ツール呼び出し詳細
- ツール実行結果
- 最終応答

## サンプル

`test/` ディレクトリに以下のサンプルがあります：

- `simple_agent/test1.py`: 基本的なエージェントの使用例
- `multi_agent/test1.py`: マルチエージェントセッションの例

## 開発者向け情報

### プロジェクト構成

```
agent_template/
├── agent/          # エージェントクラス
├── llm/            # LLMインターフェース
├── tool/           # ツールベースクラス
├── session/        # マルチエージェントセッション
├── _type/          # 型定義
├── _interface/     # インターフェース定義
└── _other/         # ユーティリティ
```

## Q&A

問題や質問がありましたら、[Issues](https://github.com/Fuji-no-yama/agent-template/issues) にてお気軽にお知らせください。