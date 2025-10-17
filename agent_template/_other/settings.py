from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    アプリ全体設定
    """

    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    embedding_model_name: str = Field("text-embedding-3-large", alias="EMBEDDING_MODEL_NAME")
    data_dir: Path = Field(
        default=Path("/workspace/data"),
        alias="DATA_DIR",
    )
    row_data_dir: Path = Field(
        default=Path("/workspace/data/row"),
        alias="ROW_DATA_DIR",
    )
    artifact_data_dir: Path = Field(
        default=Path("/workspace/data/artifact"),
        alias="ARTIFACT_DIR",
    )
    auto_classification_prompt_dir: Path = Field(
        default=Path("/workspace/code/auto_classification/prompt"),
        alias="AUTO_CLASSIFICATION_PROMPT_DIR",
    )
    auto_categorization_prompt_dir: Path = Field(
        default=Path("/workspace/code/auto_categorization/prompt"),
        alias="AUTO_CATEGORIZATION_PROMPT_DIR",
    )
    ttl_file_dir: Path = Field(
        default=Path("/workspace/data/row/20250729_日化辞資料/ttl_files"),
        alias="TTL_FILE_DIR",
    )
    data_builder_dir: Path = Field(
        default=Path("/workspace/data/artifact/data_builder"),
        alias="DATA_BUILDER_DIR",
    )
    neo4j_repository_dir: Path = Field(
        default=Path("/workspace/data/artifact/neo4j_repository"),
        alias="NEO4J_REPOSITORY_DIR",
    )
    auto_categorizer_dir: Path = Field(
        default=Path("/workspace/data/artifact/auto_categorizer"),
        alias="AUTO_CATEGORIZER_DIR",
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
