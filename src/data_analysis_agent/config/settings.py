"""Configurações globais da aplicação, lidas via pydantic-settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente/.env.

    Todas as variáveis de ambiente correspondentes usam o prefixo ``DAA_``
    (ex.: ``DAA_NIVEL_LOG``). Valores não definidos no ambiente assumem os
    defaults declarados em cada campo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DAA_",
        extra="ignore",
    )

    app_name: str = Field(
        default="AI Data Analysis Agent",
        description="Nome da aplicação.",
    )
    ambiente: str = Field(
        default="development",
        description="Ambiente de execução: 'development', 'production' ou 'test'.",
    )
    debug: bool = Field(
        default=False,
        description="Habilita o modo debug da aplicação.",
    )
    nivel_log: str = Field(
        default="INFO",
        description="Nível mínimo de log: DEBUG, INFO, WARNING, ERROR ou CRITICAL.",
    )
    diretorio_saida: Path = Field(
        default=Path("outputs"),
        description="Diretório padrão para artefatos gerados (relatórios, gráficos exportados).",
    )
    diretorio_dados: Path = Field(
        default=Path("data"),
        description="Diretório padrão para datasets de entrada.",
    )


def obter_settings() -> Settings:
    """Cria e retorna uma instância de :class:`Settings`.

    Returns:
        Settings: configurações carregadas a partir do ambiente/.env.
    """
    return Settings()
