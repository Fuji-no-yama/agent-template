from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, TypeVar, final, get_type_hints

from docstring_parser import parse

F = TypeVar("F", bound=Callable[..., Any])  # 関数型を表す型変数
R = TypeVar("R")  # 戻り値型を表す型変数


def tool(*, use_docstring: bool = True) -> Callable[[F], F]:
    """
    tool対象にしたいインスタンスメソッドにつけるデコレータ
    use_docstring が True の場合、docstringを説明文として利用する。(Falseの場合は空文字列)
    """

    def deco(func: F) -> F:
        func.__introspect_mark__ = {"use_docstring": use_docstring}
        return func

    return deco


class BaseTool:
    """
    ツールを自作するためのベースクラス
    """

    @staticmethod
    def _unwrap_callable(obj: object) -> Callable[..., Any] | None:
        """staticmethod/classmethod をアンラップして基底の関数を返す。
        呼び出し可能でない場合は None を返す。
        """
        if isinstance(obj, (staticmethod, classmethod)):
            return obj.__func__
        return obj if callable(obj) else None

    @staticmethod
    def _get_json_schema_type_mapping() -> dict[type, str]:
        """Python型からJSON Schema型への変換マッピングを返します。"""
        return {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

    @staticmethod
    def _infer_type_from_name(type_name: str) -> str:
        """型名からJSON Schema型を推測します。"""
        type_name_lower = type_name.lower()
        type_patterns = {
            ("str", "string"): "string",
            ("int", "integer"): "integer",
            ("float", "number"): "number",
            ("bool", "boolean"): "boolean",
            ("list", "array"): "array",
            ("dict", "object"): "object",
        }

        for patterns, json_type in type_patterns.items():
            if any(pattern in type_name_lower for pattern in patterns):
                return json_type
        return "string"

    @staticmethod
    def _format_annotation(ann: object) -> str:
        """型注釈をJSON Schema形式の型文字列に変換して返します。"""
        if ann is inspect.Signature.empty:
            return "string"

        type_mapping = BaseTool._get_json_schema_type_mapping()

        # 直接的な型マッチング
        if ann in type_mapping:
            return type_mapping[ann]

        # typing モジュールの型の処理
        origin = getattr(ann, "__origin__", None)
        if origin is not None:
            if origin in type_mapping:
                return type_mapping[origin]
            if origin is type(None):
                return "null"

        # 型名から推測
        type_name = getattr(ann, "__name__", str(ann))
        if isinstance(type_name, str):
            return BaseTool._infer_type_from_name(type_name)

        return "string"

    @staticmethod
    def _get_type_hints_safe(fn: Callable[..., Any]) -> dict[str, object]:
        """型ヒントを安全に取得します。取得に失敗した場合は空の dict を返します。"""
        try:
            return get_type_hints(fn, include_extras=True)
        except (NameError, TypeError, AttributeError):
            return {}

    @staticmethod
    def _parse_google_docstring_args(doc: str) -> dict[str, str]:
        """Google スタイルの docstring から引数説明を抽出して辞書を返します。

        例 (doc の一部):
            Args:
                x: 説明...
                y: 別の説明

        戻り値: {"x": "説明...", "y": "別の説明"}
        """
        ret_dict = {}
        parsed = parse(doc)
        for p in parsed.params:
            ret_dict[p.arg_name] = p.description
        return ret_dict

    @final
    def get_tool_information(self) -> list[dict]:
        """
        ツール登録されている関数(@toolがついている関数)について情報を返す関数

        Returns:
            list[dict]: ツール情報のリスト。各辞書は以下のキーを含む:
                - name (str): ツール関数の名前
                - description (str): ツール関数の説明
                - args (list[dict]): 引数情報のリスト。各辞書は以下のキーを含む:
                    - name (str): 引数名
                    - type (str): 引数の型
                    - required (bool): 引数が必須かどうか
                    - description (str): 引数の説明
                    - default (Any, optional): 引数のデフォルト値（省略可能）
        """
        tools: list[dict] = []

        for cls in type(self).mro():
            for attr_name, attr_val in cls.__dict__.items():
                fn = self._unwrap_callable(attr_val)
                if fn is None:
                    continue
                mark = getattr(fn, "__introspect_mark__", None)
                if not mark:
                    continue
                spec = self._spec_from_fn(fn)

                params = spec["parameters"]
                doc = spec["docstring"]
                short = spec.get("short_description", "")
                args_map = self._parse_google_docstring_args(doc) if doc else {}
                for param in params:
                    name = param["name"]
                    param["description"] = args_map.get(name, "")

                tool_info = {
                    "name": getattr(fn, "__name__", attr_name),
                    "description": short if mark.get("use_docstring", True) else "",
                    "args": params,
                }
                tools.append(tool_info)
        if not tools:
            err_msg = f"{type(self).__name__} にツール関数が見つかりません。"
            raise ValueError(err_msg)
        return tools

    @final
    def _spec_from_fn(self, fn: Callable[..., Any]) -> dict[str, object]:
        """指定した関数からパラメータ情報と docstring を抽出して返すヘルパー。"""
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return {"parameters": [], "docstring": ""}

        hints = self._get_type_hints_safe(fn)
        params: list[dict] = []
        empty = inspect.Signature.empty
        for name, p in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            ann = hints.get(name, p.annotation)
            required = p.default is empty
            item: dict[str, object] = {
                "name": name,
                "type": self._format_annotation(ann),
                "required": required,
                "description": "",
            }
            if not required:
                item["default"] = p.default
            params.append(item)

        doc = inspect.getdoc(fn) or ""
        # short description: first paragraph (up to first blank line)
        short = ""
        if doc:
            parts = re.split(r"\n\s*\n", doc.strip(), maxsplit=1)
            short = parts[0].strip()
        return {"parameters": params, "docstring": doc, "short_description": short}

    @final
    def has_tool(self, tool_name: str) -> bool:
        """
        指定したツール名の関数が存在するか確認する。

        Args:
            tool_name (str): 確認するツール関数の名前

        Returns:
            bool: 関数が存在する場合は True、存在しない場合は False
        """
        for cls in type(self).mro():
            for attr_name, attr_val in cls.__dict__.items():
                fn = self._unwrap_callable(attr_val)
                if fn is None:
                    continue
                if getattr(fn, "__name__", attr_name) == tool_name:
                    return True
        return False

    @final
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> R:
        """
        指定したツール名の関数を引数付きで実行し、結果を返す。

        Args:
            tool_name (str): 実行するツール関数の名前
            args (dict[str, Any]): 関数に渡す引数の辞書

        Returns:
            R: 関数の戻り値
        """
        method = getattr(self, tool_name, None)
        if method is None:
            err_msg = f"関数 '{tool_name}' が {type(self).__name__} に見つかりません。"
            raise ValueError(err_msg)
        return method(**args)


# 以下は具体的な実装例

if __name__ == "__main__":

    class MyTool(BaseTool):
        @tool(use_docstring=True)
        def add(self, x: int, y: int) -> int:
            """
            2つの整数を加算します。

            Args:
                x (int): 加算する最初の整数
                y (int): 加算する2番目の整数

            Returns:
                int: 加算結果
            """
            return x + y

        @tool(use_docstring=True)
        def multiply(self, a: float, b: float) -> float:
            """
            2つの浮動小数点数を乗算します。

            Args:
                a (float): 乗算する最初の浮動小数点数
                b (float): 乗算する2番目の浮動小数点数

            Returns:
                float: 乗算結果
            """
            return a * b

        @tool(use_docstring=True)
        def get_x(self) -> int:
            """
            xの値を取得します。

            Returns:
                int: xの値
            """
            return 42

        @tool(use_docstring=True)
        def test_all_types(self, name: str, count: int, price: float, is_active: bool, items: list, config: dict) -> str:  # noqa: FBT001, PLR0913
            """
            全ての基本型をテストします。

            Args:
                name (str): 名前文字列
                count (int): カウント整数
                price (float): 価格浮動小数点数
                is_active (bool): アクティブフラグ
                items (list): アイテムリスト
                config (dict): 設定辞書

            Returns:
                str: テスト結果
            """
            return f"Test completed: {name}, {count}, {price}, {is_active}, {len(items)}, {len(config)}"

        def divide(self, a: float, b: float) -> float:
            """
            2つの浮動小数点数を除算します。

            Args:
                a (float): 除算する最初の浮動小数点数
                b (float): 除算する2番目の浮動小数点数

            Returns:
                float: 除算結果
            """
            return a / b

    my_tool = MyTool()
    for t in my_tool.get_tool_information():
        for key in t:
            print(f"{key}: {t[key]}")

    result = my_tool.execute_tool("add", {"x": 3, "y": 5})
    print(f"Result of add: {result}")
