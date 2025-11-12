from importlib.resources import files
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    全体設定
    """

    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    embedding_model_name: str = Field("text-embedding-3-large", alias="EMBEDDING_MODEL_NAME")
    data_dir: Path = Field(
        default_factory=lambda: Path(str(files("agent_template") / "data")),
        alias="DATA_DIR",
    )
    openai_model_price: dict = Field(
        default_factory=lambda: {
            model: {"input": prices["input"] / 1_000_000, "output": prices["output"] / 1_000_000}
            for model, prices in {
                # Standard models (prices per 1M tokens)
                "gpt-5": {"input": 1.25, "output": 10.00},
                "gpt-5-mini": {"input": 0.25, "output": 2.00},
                "gpt-5-nano": {"input": 0.05, "output": 0.40},
                "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
                "gpt-5-codex": {"input": 1.25, "output": 10.00},
                "gpt-5-pro": {"input": 15.00, "output": 120.00},
                "gpt-4.1": {"input": 2.00, "output": 8.00},
                "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
                "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                "gpt-realtime": {"input": 4.00, "output": 16.00},
                "gpt-realtime-mini": {"input": 0.60, "output": 2.40},
                "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
                "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40},
                "gpt-audio": {"input": 2.50, "output": 10.00},
                "gpt-audio-mini": {"input": 0.60, "output": 2.40},
                "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
                "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
                "o1": {"input": 15.00, "output": 60.00},
                "o1-pro": {"input": 150.00, "output": 600.00},
                "o3-pro": {"input": 20.00, "output": 80.00},
                "o3": {"input": 2.00, "output": 8.00},
                "o3-deep-research": {"input": 10.00, "output": 40.00},
                "o4-mini": {"input": 1.10, "output": 4.40},
                "o4-mini-deep-research": {"input": 2.00, "output": 8.00},
                "o3-mini": {"input": 1.10, "output": 4.40},
                "o1-mini": {"input": 1.10, "output": 4.40},
                "codex-mini-latest": {"input": 1.50, "output": 6.00},
                "gpt-5-search-api": {"input": 1.25, "output": 10.00},
                "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60},
                "gpt-4o-search-preview": {"input": 2.50, "output": 10.00},
                "computer-use-preview": {"input": 3.00, "output": 12.00},
                "gpt-image-1": {"input": 5.00, "output": 0.00},
                "gpt-image-1-mini": {"input": 2.00, "output": 0.00},
                # Legacy models (prices per 1M tokens)
                "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
                "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
                "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
                "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
                "gpt-4-1106-vision-preview": {"input": 10.00, "output": 30.00},
                "gpt-4-0613": {"input": 30.00, "output": 60.00},
                "gpt-4-0314": {"input": 30.00, "output": 60.00},
                "gpt-4-32k": {"input": 60.00, "output": 120.00},
                "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
                "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
                "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
                "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
                "gpt-3.5-0301": {"input": 1.50, "output": 2.00},
                "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
                "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
                "davinci-002": {"input": 2.00, "output": 2.00},
                "babbage-002": {"input": 0.40, "output": 0.40},
            }.items()
        },
        alias="OPENAI_MODEL_PRICE",
    )

    # .env > init kwargs > OS env の優先順位を維持
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["Settings"],  # noqa: ARG003
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return dotenv_settings, init_settings, env_settings, file_secret_settings


settings = Settings()
