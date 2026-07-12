"""Modelos de domínio relacionados a gráficos e metadados de relatório.

Apenas estrutura — sem lógica de geração de gráficos ou relatórios implementada.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TipoGrafico(str, Enum):
    """Tipos de gráfico suportados pelo módulo de visualização."""

    LINHA = "line"
    DISPERSAO = "scatter"
    BARRA = "bar"
    BOXPLOT = "boxplot"


class ChartSpec(BaseModel):
    """Especificação de um gráfico gerado, com a figura Plotly serializada.

    Conforme a Seção 7 do SDD v2.0, a figura nunca é guardada como objeto
    ``plotly.graph_objects.Figure`` — é serializada via ``fig.to_dict()``
    para manter os modelos de domínio livres de dependências externas em
    tempo de execução. Quem precisa renderizar reconstrói o objeto Plotly
    a partir deste dict no momento da exibição.
    """

    titulo: str = Field(..., description="Título do gráfico.")
    tipo: TipoGrafico = Field(..., description="Tipo do gráfico.")
    figura: dict[str, Any] = Field(
        ..., description="Figura Plotly serializada (resultado de `fig.to_dict()`)."
    )
    descricao: str | None = Field(
        default=None,
        description="Descrição ou insight em linguagem natural associado ao gráfico.",
    )


class ReportMetadata(BaseModel):
    """Metadados associados a um relatório gerado."""

    titulo: str = Field(..., description="Título do relatório.")
    fonte_dados: str = Field(
        ..., description="Identificação da fonte de dados que originou o relatório."
    )
    gerado_em: datetime = Field(
        default_factory=datetime.now, description="Data e hora de geração do relatório."
    )
    autor: str | None = Field(
        default=None, description="Autor ou sistema responsável pela geração do relatório."
    )
    secoes: list[str] = Field(
        default_factory=list, description="Lista de seções presentes no relatório."
    )
