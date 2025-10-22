class RetryableError(Exception):
    """一時的な失敗（リトライ可能）を表す例外"""

    def __init__(self, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception
