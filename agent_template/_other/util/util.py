import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


def get_logger(
    log_dir: os.PathLike,
    suppress_warning_lib_list: list[str] | None = None,
    log_level: int = logging.INFO,
    file_prefix: str = "",
) -> logging.Logger:
    """
    指定されたディレクトリにログを出力するロガーを作成する

    Args:
        log_dir (os.PathLike): ログの出力先ディレクトリ。
        suppress_warning_lib_list (Optional[List[str]]): ロガーの出力を抑制するライブラリのリスト。
        log_level (int): ロガーの出力レベル。logging.INFO, logging.DEBUGなど。デフォルトはlogging.INFO。
        file_prefix (str): ログファイル名の接頭辞。デフォルトは空文字列。

    Returns:
        logging.Logger: 設定済みのloggerオブジェクト。
    """
    if suppress_warning_lib_list is None:
        suppress_warning_lib_list = []

    for lib_name in suppress_warning_lib_list:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    now_jst = datetime.now(timezone(timedelta(hours=9)))
    time_stamp = now_jst.strftime("%Y%m%d_%H%M_%S")
    logger_name = f"{__name__}.{str(log_dir)}.{time_stamp}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if logger.hasHandlers():  # 複数回呼ばれた際にハンドラーが重複しないように、既存のハンドラーをクリア
        logger.handlers.clear()

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)  # ディレクトリが存在しない場合は作成
    log_file_path = log_dir_path / f"{file_prefix}_{time_stamp}.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)  # ファイルハンドラーのレベルも設定
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
