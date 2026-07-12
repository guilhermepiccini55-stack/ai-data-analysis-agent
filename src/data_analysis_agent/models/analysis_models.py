"""Modelos de domínio relacionados à análise estatística dos dados.

Apenas estrutura — sem lógica de análise estatística implementada.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.report_models import ChartSpec, ReportMetadata


class MetodoOutlier(str, Enum):
    """Métodos suportados para detecção de outliers."""

    IQR = "iqr"
    ZSCORE = "zscore"


class OutlierResult(BaseModel):
    """Resultado da detecção de outliers em uma coluna."""

    coluna: str = Field(..., description="Nome da coluna analisada.")
    metodo: MetodoOutlier = Field(..., description="Método utilizado na detecção.")
    limite_inferior: float = Field(..., description="Limite inferior considerado para outliers.")
    limite_superior: float = Field(..., description="Limite superior considerado para outliers.")
    indices_outliers: list[int] = Field(
        default_factory=list, description="Índices das linhas identificadas como outliers."
    )
    total_outliers: int = Field(..., description="Quantidade total de outliers identificados.")


class CorrelationResult(BaseModel):
    """Resultado da análise de correlação entre variáveis numéricas."""

    metodo: str = Field(
        ..., description="Método de correlação utilizado (ex.: 'pearson', 'spearman')."
    )
    matriz_correlacao: dict[str, dict[str, float]] = Field(
        ..., description="Matriz de correlação entre colunas numéricas."
    )
    pares_fortemente_correlacionados: list[tuple[str, str, float]] = Field(
        default_factory=list,
        description="Pares de colunas com correlação acima do limiar configurado.",
    )


class AnalysisSummary(BaseModel):
    """Resumo estruturado dos resultados da etapa de análise estatística.

    Agrega os artefatos produzidos exclusivamente por ``analysis.py``,
    desempenhando para a etapa de análise o mesmo papel que
    ``CleaningReport`` desempenha para a etapa de limpeza.
    """

    estatisticas_descritivas: dict[str, Any] = Field(
        default_factory=dict, description="Estatísticas descritivas calculadas sobre o dataset."
    )
    outliers: list[OutlierResult] = Field(
        default_factory=list, description="Resultados de detecção de outliers por coluna."
    )
    correlacoes: CorrelationResult | None = Field(
        default=None, description="Resultado da análise de correlação, se aplicável."
    )


class AnalysisResult(BaseModel):
    """Resultado completo e autossuficiente de uma execução do pipeline de análise.

    Conforme a Seção 5 do SDD v2.0, este é o único objeto devolvido pela
    ``AnalysisEngine`` (stateless): contém tudo que uma interface precisa
    para exibir resultados, sem depender de estado guardado na engine.
    """

    fonte_dados: str = Field(
        ..., description="Identificação da fonte de dados analisada (ex.: caminho do arquivo)."
    )
    relatorio_limpeza: CleaningReport = Field(
        ..., description="Relatório da etapa de limpeza de dados."
    )
    resultado_analise: AnalysisSummary = Field(
        ..., description="Resumo estruturado da etapa de análise estatística."
    )
    graficos: list[ChartSpec] = Field(
        default_factory=list, description="Especificações dos gráficos gerados."
    )
    metadados_relatorio: ReportMetadata | None = Field(
        default=None, description="Metadados do relatório final, se gerado."
    )
    gerado_em: datetime = Field(
        default_factory=datetime.now, description="Momento em que o resultado foi gerado."
    )