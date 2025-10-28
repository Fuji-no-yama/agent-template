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
