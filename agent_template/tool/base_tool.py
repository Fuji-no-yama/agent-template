from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, TypeVar, final, get_type_hints

F = TypeVar("F", bound=Callable[..., Any])


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
    def _format_annotation(ann: object) -> str:
        """型注釈を読みやすい文字列に整形して返します。"""
        if ann is inspect.Signature.empty:
            return "unannotated"
        mod = getattr(ann, "__module__", "")
        if mod == "typing":
            return str(ann).replace("typing.", "")
        return ann.__name__ if hasattr(ann, "__name__") else str(ann)

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
        if not doc:
            return {}

        # extract the Args: block first
        m_block = re.search(r"^\s*Args:\s*\n([\s\S]*?)(?=\n\s*\w+:|\Z)", doc, re.MULTILINE)
        if not m_block:
            return {}
        block = m_block.group(1)

        # match each parameter and its (possibly multiline) description
        param_re = re.compile(
            r"^\s{4,}([A-Za-z0-9_]+)(?:\s*\([^\)]*\))?\s*:\s*(.*?)(?=^\s{4,}[A-Za-z0-9_]+(?:\s*\([^\)]*\))?\s*:|\Z)",
            re.MULTILINE | re.DOTALL,
        )

        res: dict[str, str] = {}
        for m in param_re.finditer(block):
            name = m.group(1)
            desc = m.group(2)
            # normalize whitespace and strip
            desc = re.sub(r"\s+", " ", desc).strip()
            res[name] = desc
        return res

    @final
    def get_tool_information(self) -> str:
        """
        ツール登録されている関数(@toolがついている関数)について以下の形式の情報をListとして返す関数
        {
            "name": 関数名,
            "description": 関数の説明文,
            "args": [
                {
                    "name": 引数名,
                    "description": 引数の説明文,
                    "type": 引数の型,
                    "required": True or False,
                }, ...
            ]
        }
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
            return "{}"
        return tools[0] if len(tools) == 1 else tools

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
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """
        指定したツール名の関数を引数付きで実行し、結果を返す。
        """


# 以下は具体的な実装例


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
