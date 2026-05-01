"""アプリ全体の設定。

`.env` から読み、型付きで提供する。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """環境変数ベースのアプリケーション設定。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model_fast: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_FAST")
    openai_model_smart: str = Field(default="gpt-4o", alias="OPENAI_MODEL_SMART")

    # Tavily
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # 動作設定
    log_level: str = Field(default="INFO", alias="DAS_LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), alias="DAS_DATA_DIR")
    linking_threshold: float = Field(default=0.6, alias="DAS_LINKING_THRESHOLD")

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


_settings: Settings | None = None


def get_settings() -> Settings:
    """シングルトンに近い形で Settings を返す。テストでは reset_settings() で破棄できる。"""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """主にテスト用。次回 get_settings() で再ロードされる。"""

    global _settings
    _settings = None
