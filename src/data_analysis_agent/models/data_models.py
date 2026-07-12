"""Modelos de domínio relacionados à limpeza e ao perfilamento de dados.

Apenas estrutura — sem lógica de limpeza ou perfilamento implementada.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Perfil descritivo de uma coluna do dataset."""

    nome: str = Field(..., description="Nome da coluna.")
    tipo_inferido: str = Field(
        ..., description="Tipo de dado inferido (ex.: 'numeric', 'date', 'categorical')."
    )
    tipo_original: str = Field(..., description="Dtype original do pandas antes da inferência.")
    total_valores: int = Field(..., description="Número total de valores na coluna.")
    valores_ausentes: int = Field(..., description="Quantidade de valores ausentes (NaN/None).")
    percentual_ausente: float = Field(
        ..., description="Percentual de valores ausentes em relação ao total."
    )
    valores_unicos: int = Field(..., description="Quantidade de valores distintos na coluna.")
    amostra_valores: list[Any] = Field(
        default_factory=list, description="Amostra de valores representativos da coluna."
    )


class CleaningReport(BaseModel):
    """Relatório resultante da etapa de limpeza de dados."""

    formato_original: tuple[int, int] = Field(
        ..., description="Formato (linhas, colunas) do dataset antes da limpeza."
    )
    formato_final: tuple[int, int] = Field(
        ..., description="Formato (linhas, colunas) do dataset após a limpeza."
    )
    duplicatas_removidas: int = Field(..., description="Número de linhas duplicadas removidas.")
    colunas_perfil: list[ColumnProfile] = Field(
        default_factory=list, description="Perfil individual de cada coluna do dataset."
    )
    valores_imputados: dict[str, int] = Field(
        default_factory=dict, description="Quantidade de valores imputados por coluna."
    )
    conversoes_tipo: dict[str, str] = Field(
        default_factory=dict,
        description="Conversões de tipo aplicadas por coluna (tipo original -> tipo final).",
    )
    gerado_em: datetime = Field(
        default_factory=datetime.now, description="Momento em que o relatório foi gerado."
    )
