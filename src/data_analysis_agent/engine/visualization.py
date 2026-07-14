"""Geração de visualizações Plotly (engine.visualization).

Conforme a Seção 7 do SDD v2.0, nenhuma figura é guardada como objeto
``plotly.graph_objects.Figure`` nos modelos de domínio — apenas como
``dict`` serializável, obtido via ``json.loads(fig.to_json())``.

A API pública deste módulo é intencionalmente mínima e recebe, além do
``DataFrame``, o ``AnalysisSummary`` já produzido pela etapa de análise
(``engine.analysis.analisar_dados``). Isso evita duas coisas:

- recomputar detecção de outliers/correlações dentro da camada de
  visualização (duplicação de lógica e risco de o gráfico divergir do
  relatório);
- que este módulo tome decisões estatísticas próprias — ele apenas
  representa visualmente artefatos que a análise já produziu.

Escopo do primeiro incremento (decisão registrada, não assumida):
apenas boxplot (por coluna presente em ``resumo_analise.outliers``) e
dispersão (por par presente em
``resumo_analise.correlacoes.pares_fortemente_correlacionados``).
Histograma, barra e linha ficam fora deste incremento porque
``AnalysisSummary`` ainda não contém informação de distribuição ou de
frequência categórica.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from data_analysis_agent.exceptions.errors import VisualizationError
from data_analysis_agent.models.analysis_models import (
    AnalysisSummary,
    CorrelationResult,
    OutlierResult,
)
from data_analysis_agent.models.report_models import ChartSpec, TipoGrafico


def gerar_visualizacoes(
    dados: pd.DataFrame,
    resumo_analise: AnalysisSummary,
) -> list[ChartSpec]:
    """Gera as visualizações correspondentes a um resumo de análise já produzido.

    Args:
        dados: ``DataFrame`` a partir do qual os gráficos são construídos.
            Deve ser o mesmo (ou equivalente) DataFrame que originou
            ``resumo_analise``.
        resumo_analise: resultado já produzido por
            ``engine.analysis.analisar_dados``, usado para decidir quais
            gráficos gerar sem recomputar estatísticas.

    Returns:
        list[ChartSpec]: especificações dos gráficos gerados, com a
        figura Plotly serializada como dict. Lista vazia se não houver
        dados, outliers ou correlações fortes a representar.

    Raises:
        VisualizationError: se ``dados`` não for um ``pandas.DataFrame``.
    """
    _validar_tipo_dataframe(dados)

    if dados.empty:
        return []

    graficos: list[ChartSpec] = []
    graficos.extend(_gerar_boxplots(dados, resumo_analise.outliers))
    graficos.extend(_gerar_dispersoes(dados, resumo_analise.correlacoes))
    return graficos


def _validar_tipo_dataframe(dados: pd.DataFrame) -> None:
    """Garante que ``dados`` é um ``pandas.DataFrame`` antes de prosseguir.

    Raises:
        VisualizationError: se o tipo recebido não for ``pd.DataFrame``.
    """
    if not isinstance(dados, pd.DataFrame):
        raise VisualizationError(
            f"Esperado um pandas.DataFrame, recebido {type(dados).__name__}."
        )


def _gerar_boxplots(
    dados: pd.DataFrame,
    outliers: list[OutlierResult],
) -> list[ChartSpec]:
    """Gera um boxplot para cada coluna presente em ``outliers``.

    Colunas ausentes de ``dados`` são ignoradas silenciosamente — não é
    responsabilidade desta camada validar a consistência entre
    ``resumo_analise`` e ``dados`` (isso é uma invariante do chamador,
    a ``AnalysisEngine``).
    """
    return [
        _gerar_boxplot(dados, outlier)
        for outlier in outliers
        if outlier.coluna in dados.columns
    ]


def _gerar_boxplot(dados: pd.DataFrame, outlier: OutlierResult) -> ChartSpec:
    """Constrói o ``ChartSpec`` de um boxplot para uma coluna analisada."""
    coluna = outlier.coluna
    fig = go.Figure(
        data=[go.Box(y=dados[coluna].dropna(), name=coluna, boxpoints="outliers")]
    )
    fig.update_layout(title=f"Distribuição de {coluna} (boxplot)")

    descricao = (
        f"{outlier.total_outliers} outlier(s) detectado(s) via "
        f"{outlier.metodo.value.upper()}."
        if outlier.total_outliers > 0
        else f"Nenhum outlier detectado via {outlier.metodo.value.upper()}."
    )

    return ChartSpec(
        titulo=f"Distribuição de {coluna} (boxplot)",
        tipo=TipoGrafico.BOXPLOT,
        figura=_serializar_figura(fig),
        descricao=descricao,
    )


def _gerar_dispersoes(
    dados: pd.DataFrame,
    correlacoes: CorrelationResult | None,
) -> list[ChartSpec]:
    """Gera um gráfico de dispersão para cada par fortemente correlacionado.

    Retorna lista vazia se não houver resultado de correlação ou se
    nenhum par tiver sido considerado fortemente correlacionado.
    """
    if correlacoes is None:
        return []

    return [
        _gerar_dispersao(dados, coluna_x, coluna_y, valor, correlacoes.metodo)
        for coluna_x, coluna_y, valor in correlacoes.pares_fortemente_correlacionados
        if coluna_x in dados.columns and coluna_y in dados.columns
    ]


def _gerar_dispersao(
    dados: pd.DataFrame,
    coluna_x: str,
    coluna_y: str,
    valor_correlacao: float,
    metodo: str,
) -> ChartSpec:
    """Constrói o ``ChartSpec`` de uma dispersão entre duas colunas correlacionadas."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=dados[coluna_x],
                y=dados[coluna_y],
                mode="markers",
                name=f"{coluna_x} vs {coluna_y}",
            )
        ]
    )
    titulo = f"{coluna_x} vs {coluna_y} (r={valor_correlacao:.2f})"
    fig.update_layout(title=titulo, xaxis_title=coluna_x, yaxis_title=coluna_y)

    return ChartSpec(
        titulo=titulo,
        tipo=TipoGrafico.DISPERSAO,
        figura=_serializar_figura(fig),
        descricao=(
            f"Correlação {metodo} forte entre {coluna_x} e {coluna_y} "
            f"(r={valor_correlacao:.2f})."
        ),
    )


def _serializar_figura(fig: go.Figure) -> dict[str, Any]:
    """Serializa uma figura Plotly em um dict puro, apto a virar JSON.

    Usa ``json.loads(fig.to_json())`` em vez de ``fig.to_dict()``: o
    encoder JSON do Plotly já converte arrays/escalares NumPy em tipos
    nativos, garantindo que o dict resultante seja serializável sem
    depender de uma sanitização recursiva própria — mesma exigência de
    tipos nativos já aplicada em ``AnalysisSummary``/``AnalysisResult``.
    """
    return json.loads(fig.to_json())