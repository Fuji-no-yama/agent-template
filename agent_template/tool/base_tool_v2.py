from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal, TypeVar, final, get_args, get_origin, get_type_hints

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
    ツールを自作するためのベースクラス（v2拡張版）
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
    def _get_basic_type_mapping() -> dict[type, str]:
        """基本型からJSON Schema型への変換マッピングを返します。"""
        return {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

    @staticmethod
    def _analyze_type_annotation(annotation: object) -> dict[str, Any]:
        """型注釈を詳細に解析してJSON Schema形式の情報を返します."""
        if annotation is inspect.Signature.empty or annotation is None:
            return {"type": "string", "additional_info": "no_annotation"}

        basic_mapping = BaseTool._get_basic_type_mapping()

        # 基本型の直接マッチング
        if annotation in basic_mapping:
            return {"type": basic_mapping[annotation], "additional_info": f"basic_type_{annotation.__name__}"}

        # typing モジュールの型解析
        return BaseTool._analyze_complex_type(annotation)

    @staticmethod
    def _analyze_complex_type(annotation: object) -> dict[str, Any]:
        """複雑な型注釈を解析します."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Literal:
            return BaseTool._handle_literal_type(args)

        if BaseTool._is_union_type(origin):
            return BaseTool._handle_union_type(args, annotation)

        if origin is list:
            return BaseTool._handle_list_type(args, annotation)

        if origin is dict:
            return BaseTool._handle_dict_type(args, annotation)

        return BaseTool._handle_unsupported_type(annotation, origin)

    @staticmethod
    def _handle_literal_type(args: tuple[Any, ...]) -> dict[str, Any]:
        """Literal型を処理します."""
        return {
            "type": "string",
            "enum": list(args),
            "additional_info": f"literal_{args}",
        }

    @staticmethod
    def _is_union_type(origin: object) -> bool:
        """Union型かどうかを判定します."""
        return origin is type(None) or str(origin) == "typing.Union"

    @staticmethod
    def _handle_union_type(args: tuple[Any, ...], annotation: object) -> dict[str, Any]:
        """Union型を処理します."""
        optional_args_count = 2
        if len(args) == optional_args_count and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            result = BaseTool._analyze_type_annotation(non_none_type)
            result["nullable"] = True
            result["additional_info"] = f"optional_{result.get('additional_info', '')}"
            return result

        msg = f"複雑なUnion型は未サポートです: {annotation}"
        raise ValueError(msg)

    @staticmethod
    def _handle_list_type(args: tuple[Any, ...], annotation: object) -> dict[str, Any]:
        """list型を処理します."""
        if not args:
            return {"type": "array", "additional_info": "list_no_args"}

        if len(args) == 1:
            item_type = BaseTool._analyze_type_annotation(args[0])
            return {
                "type": "array",
                "items": item_type,
                "additional_info": f"list_{args[0]}",
            }

        msg = f"複雑なlist型は未サポートです: {annotation}"
        raise ValueError(msg)

    @staticmethod
    def _handle_dict_type(args: tuple[Any, ...], annotation: object) -> dict[str, Any]:
        """dict型を処理します."""
        if not args:
            return {"type": "object", "additional_info": "dict_no_args"}

        dict_args_count = 2
        if len(args) == dict_args_count:
            key_type, value_type = args
            if key_type is not str:
                msg = f"dict のキー型は str のみサポートです: {key_type}"
                raise ValueError(msg)

            value_type_info = BaseTool._analyze_type_annotation(value_type)
            return {
                "type": "object",
                "additionalProperties": value_type_info,
                "additional_info": f"dict_{key_type}_{value_type}",
            }

        msg = f"複雑なdict型は未サポートです: {annotation}"
        raise ValueError(msg)

    @staticmethod
    def _handle_unsupported_type(annotation: object, origin: object) -> dict[str, Any]:
        """未サポート型を処理します."""
        if origin is not None:
            msg = f"未サポートの型です: {annotation} (origin: {origin})"
            raise ValueError(msg)

        if hasattr(annotation, "__name__"):
            msg = f"カスタムクラス型は未サポートです: {annotation.__name__}"
            raise ValueError(msg)

        return {"type": "string", "additional_info": f"fallback_{str(annotation)}"}

    @staticmethod
    def _get_type_hints_safe(fn: Callable[..., Any]) -> dict[str, object]:
        """型ヒントを安全に取得します。取得に失敗した場合は空の dict を返します。"""
        try:
            return get_type_hints(fn, include_extras=True)
        except (NameError, TypeError, AttributeError):
            return {}

    @staticmethod
    def _parse_google_docstring(doc: str) -> dict[str, Any]:
        """
        Google スタイルの docstring から情報を抽出します。

        Returns:
            dict: 以下のキーを含む辞書
                - description (str): 関数の説明
                - args (dict): 引数情報 {arg_name: {"description": str, "type_name": str}}
                - returns (dict): 戻り値情報 {"description": str, "type_name": str}
        """
        result = {
            "description": "",
            "args": {},
            "returns": {"description": "", "type_name": ""},
        }

        if not doc:
            return result

        parsed = parse(doc)

        # 関数の説明（短い説明）
        if parsed.short_description:
            result["description"] = parsed.short_description
        elif parsed.long_description:
            result["description"] = parsed.long_description

        # 引数情報
        for param in parsed.params:
            result["args"][param.arg_name] = {
                "description": param.description or "",
                "type_name": param.type_name or "",
            }

        # 戻り値情報
        if parsed.returns:
            result["returns"] = {
                "description": parsed.returns.description or "",
                "type_name": parsed.returns.type_name or "",
            }

        return result

    @final
    def get_tool_information(self) -> list[dict]:
        """
        ツール登録されている関数(@toolがついている関数)について詳細情報を返す関数

        Returns:
            list[dict]: ツール情報のリスト。各辞書は以下のキーを含む:
                - name (str): ツール関数の名前
                - description (str): ツール関数の説明
                - args (list[dict]): 引数情報のリスト。各辞書は以下のキーを含む:
                    - name (str): 引数名
                    - type_info (dict): 詳細な型情報 <- ここに諸々の情報が入る
                    - required (bool): 引数が必須かどうか
                    - description (str): 引数の説明
                    - default (Any, optional): 引数のデフォルト値（省略可能）
                    - docstring_type (str): docstring内の型名
                - returns (dict): 戻り値情報
                    - type_info (dict): 戻り値の型情報
                    - description (str): 戻り値の説明
                    - docstring_type (str): docstring内の戻り値型名
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

                try:
                    spec = self._spec_from_fn(fn)
                    tools.append(spec)
                except ValueError as e:
                    # 未サポート型がある場合はエラーを発生
                    func_name = getattr(fn, "__name__", attr_name)
                    msg = f"関数 '{func_name}' で未サポート型が検出されました: {e}"
                    raise ValueError(msg) from e

        if not tools:
            err_msg = f"{type(self).__name__} にツール関数が見つかりません。"
            raise ValueError(err_msg)
        return tools

    @final
    def _spec_from_fn(self, fn: Callable[..., Any]) -> dict[str, Any]:
        """指定した関数からパラメータ情報と docstring を詳細に抽出して返すヘルパー。"""
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return {
                "name": getattr(fn, "__name__", "unknown"),
                "description": "",
                "args": [],
                "returns": {"type_info": {"type": "unknown"}, "description": "", "docstring_type": ""},
            }

        # Type hints取得
        hints = self._get_type_hints_safe(fn)

        # Docstring解析
        doc = inspect.getdoc(fn) or ""
        docstring_info = self._parse_google_docstring(doc)

        # 引数情報の構築
        args_info: list[dict] = []
        empty = inspect.Signature.empty

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            # 型注釈の取得と解析
            annotation = hints.get(name, param.annotation)
            try:
                type_info = self._analyze_type_annotation(annotation)
            except ValueError as e:
                # 未サポート型の場合はエラーを再発生
                msg = f"引数 '{name}' の型が未サポートです: {e}"
                raise ValueError(msg) from e

            # docstringからの情報取得
            docstring_arg_info = docstring_info["args"].get(name, {})

            arg_info = {
                "name": name,
                "type_info": type_info,
                "required": param.default is empty,
                "description": docstring_arg_info.get("description", ""),
                "docstring_type": docstring_arg_info.get("type_name", ""),
            }

            if param.default is not empty:
                arg_info["default"] = param.default

            args_info.append(arg_info)

        # 戻り値情報の構築
        return_annotation = hints.get("return", sig.return_annotation)
        try:
            return_type_info = self._analyze_type_annotation(return_annotation)
        except ValueError as e:
            msg = f"戻り値の型が未サポートです: {e}"
            raise ValueError(msg) from e

        return_info = {
            "type_info": return_type_info,
            "description": docstring_info["returns"]["description"],
            "docstring_type": docstring_info["returns"]["type_name"],
        }

        return {
            "name": getattr(fn, "__name__", "unknown"),
            "description": docstring_info["description"],
            "args": args_info,
            "returns": return_info,
        }

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

    class MyToolV2(BaseTool):
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
        def process_list(self, items: list[str], count: int | None = None) -> list[str]:
            """
            文字列リストを処理します。

            Args:
                items (list[str]): 処理対象の文字列リスト
                count (Optional[int]): 処理する最大件数

            Returns:
                list[str]: 処理済みの文字列リスト
            """
            result = [item.upper() for item in items]
            if count is not None:
                result = result[:count]
            return result

        @tool(use_docstring=True)
        def select_option(self, mode: Literal["fast", "normal", "slow"]) -> str:
            """
            動作モードを選択します。

            Args:
                mode (Literal["fast", "normal", "slow"]): 動作モード

            Returns:
                str: 選択されたモードの説明
            """
            modes = {
                "fast": "高速モード",
                "normal": "通常モード",
                "slow": "低速モード",
            }
            return modes[mode]

        @tool(use_docstring=True)
        def process_config(self, config: dict[str, int]) -> dict[str, str]:
            """
            設定辞書を処理します。

            Args:
                config (dict[str, int]): 設定値の辞書

            Returns:
                dict[str, str]: 処理済み設定辞書
            """
            return {k: f"processed_{v}" for k, v in config.items()}

        @tool(use_docstring=True)
        def process_data(self, data: list[dict[str, int]]) -> str:
            pass

    my_tool = MyToolV2()

    print("=== ツール情報の詳細表示 ===")
    tools = my_tool.get_tool_information()

    for tool_info in tools:
        print(f"\n関数名: {tool_info['name']}")
        print(f"説明: {tool_info['description']}")

        print("\n引数:")
        for arg in tool_info["args"]:
            print(f"  - {arg['name']} ({arg['type_info']})")
            print(f"    説明: {arg['description']}")
            print(f"    必須: {arg['required']}")
            if "default" in arg:
                print(f"    デフォルト: {arg['default']}")
            if arg["docstring_type"]:
                print(f"    docstring型: {arg['docstring_type']}")

        print(f"\n戻り値: {tool_info['returns']['type_info']}")
        if tool_info["returns"]["description"]:
            print(f"戻り値説明: {tool_info['returns']['description']}")
        if tool_info["returns"]["docstring_type"]:
            print(f"戻り値docstring型: {tool_info['returns']['docstring_type']}")

        print("-" * 50)

    # 実行テスト
    print("\n=== 実行テスト ===")
    result1 = my_tool.execute_tool("add", {"x": 3, "y": 5})
    print(f"add(3, 5) = {result1}")

    result2 = my_tool.execute_tool("process_list", {"items": ["hello", "world"], "count": 1})
    print(f"process_list(['hello', 'world'], count=1) = {result2}")

    result3 = my_tool.execute_tool("select_option", {"mode": "fast"})
    print(f"select_option('fast') = {result3}")
