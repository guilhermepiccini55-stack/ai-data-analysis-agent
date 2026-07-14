"""Testes unitários de engine.visualization.

Testes black-box: cobrem apenas a API pública (`gerar_visualizacoes`),
sem depender de detalhes de implementação das funções privadas.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data_analysis_agent.engine.visualization import gerar_visualizacoes
from data_analysis_agent.exceptions.errors import VisualizationError
from data_analysis_agent.models.analysis_models import (
    AnalysisSummary,
    CorrelationResult,
    MetodoOutlier,
    OutlierResult,
)
from data_analysis_agent.models.report_models import ChartSpec, TipoGrafico


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dados_numericos() -> pd.DataFrame:
    """DataFrame com duas colunas numéricas, uma delas com um valor discrepante."""
    return pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [10, 20, 30, 40, 50]})


@pytest.fixture
def outlier_coluna_a() -> OutlierResult:
    """Resultado de detecção de outliers para a coluna 'a', com 1 outlier."""
    return OutlierResult(
        coluna="a",
        metodo=MetodoOutlier.IQR,
        limite_inferior=-5.0,
        limite_superior=10.0,
        indices_outliers=[4],
        total_outliers=1,
    )


@pytest.fixture
def outlier_coluna_b_sem_outliers() -> OutlierResult:
    """Resultado de detecção de outliers para a coluna 'b', sem outliers."""
    return OutlierResult(
        coluna="b",
        metodo=MetodoOutlier.IQR,
        limite_inferior=0.0,
        limite_superior=60.0,
        indices_outliers=[],
        total_outliers=0,
    )


@pytest.fixture
def correlacao_forte_a_b() -> CorrelationResult:
    """Resultado de correlação com um único par fortemente correlacionado."""
    return CorrelationResult(
        metodo="pearson",
        matriz_correlacao={"a": {"a": 1.0, "b": 0.95}, "b": {"a": 0.95, "b": 1.0}},
        pares_fortemente_correlacionados=[("a", "b", 0.95)],
    )


@pytest.fixture
def resumo_vazio() -> AnalysisSummary:
    """AnalysisSummary sem outliers e sem correlações."""
    return AnalysisSummary()


# ---------------------------------------------------------------------------
# DataFrame vazio
# ---------------------------------------------------------------------------


class TestDataFrameVazio:
    """Casos em que a entrada de dados não contém linhas."""

    def test_retorna_lista_vazia_para_dataframe_vazio(
        self, resumo_vazio: AnalysisSummary
    ) -> None:
        dados_vazios = pd.DataFrame()

        resultado = gerar_visualizacoes(dados_vazios, resumo_vazio)

        assert resultado == []

    def test_retorna_lista_vazia_para_dataframe_vazio_mesmo_com_outliers(
        self, outlier_coluna_a: OutlierResult
    ) -> None:
        dados_vazios = pd.DataFrame()
        resumo = AnalysisSummary(outliers=[outlier_coluna_a])

        resultado = gerar_visualizacoes(dados_vazios, resumo)

        assert resultado == []


# ---------------------------------------------------------------------------
# Entrada inválida
# ---------------------------------------------------------------------------


class TestEntradaInvalida:
    """Casos em que o tipo de `dados` não é um DataFrame."""

    def test_levanta_visualization_error_para_tipo_invalido(
        self, resumo_vazio: AnalysisSummary
    ) -> None:
        with pytest.raises(VisualizationError):
            gerar_visualizacoes("não é um dataframe", resumo_vazio)

    def test_levanta_visualization_error_para_none(
        self, resumo_vazio: AnalysisSummary
    ) -> None:
        with pytest.raises(VisualizationError):
            gerar_visualizacoes(None, resumo_vazio)


# ---------------------------------------------------------------------------
# Geração de boxplot
# ---------------------------------------------------------------------------


class TestGeracaoDeBoxplot:
    """Casos de geração de boxplot a partir de `AnalysisSummary.outliers`."""

    def test_gera_um_boxplot_quando_existe_outlier_result(
        self, dados_numericos: pd.DataFrame, outlier_coluna_a: OutlierResult
    ) -> None:
        resumo = AnalysisSummary(outliers=[outlier_coluna_a])

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert len(resultado) == 1
        assert resultado[0].tipo == TipoGrafico.BOXPLOT

    def test_gera_um_boxplot_por_coluna_presente_em_outliers(
        self,
        dados_numericos: pd.DataFrame,
        outlier_coluna_a: OutlierResult,
        outlier_coluna_b_sem_outliers: OutlierResult,
    ) -> None:
        resumo = AnalysisSummary(
            outliers=[outlier_coluna_a, outlier_coluna_b_sem_outliers]
        )

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert len(resultado) == 2
        assert all(g.tipo == TipoGrafico.BOXPLOT for g in resultado)

    def test_nao_gera_boxplot_para_colunas_ausentes_de_outliers(
        self, dados_numericos: pd.DataFrame, outlier_coluna_a: OutlierResult
    ) -> None:
        """Regra oficial do projeto: boxplot só para colunas presentes em
        `outliers` — não para todas as colunas numéricas do DataFrame."""
        resumo = AnalysisSummary(outliers=[outlier_coluna_a])

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        # dados_numericos tem 2 colunas numéricas ("a" e "b"), mas apenas
        # "a" está presente em `outliers`.
        assert len(resultado) == 1

    def test_retorna_lista_vazia_quando_nao_ha_outliers(
        self, dados_numericos: pd.DataFrame, resumo_vazio: AnalysisSummary
    ) -> None:
        resultado = gerar_visualizacoes(dados_numericos, resumo_vazio)

        assert resultado == []


# ---------------------------------------------------------------------------
# Geração de dispersão
# ---------------------------------------------------------------------------


class TestGeracaoDeDispersao:
    """Casos de geração de dispersão a partir de correlações fortes."""

    def test_gera_dispersao_quando_existe_correlacao_forte(
        self,
        dados_numericos: pd.DataFrame,
        correlacao_forte_a_b: CorrelationResult,
    ) -> None:
        resumo = AnalysisSummary(outliers=[], correlacoes=correlacao_forte_a_b)

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert len(resultado) == 1
        assert resultado[0].tipo == TipoGrafico.DISPERSAO

    def test_retorna_lista_vazia_quando_correlacoes_e_none(
        self, dados_numericos: pd.DataFrame
    ) -> None:
        resumo = AnalysisSummary(outliers=[], correlacoes=None)

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert resultado == []

    def test_retorna_lista_vazia_quando_nao_ha_pares_fortemente_correlacionados(
        self, dados_numericos: pd.DataFrame
    ) -> None:
        correlacao_sem_pares_fortes = CorrelationResult(
            metodo="pearson",
            matriz_correlacao={"a": {"a": 1.0, "b": 0.1}, "b": {"a": 0.1, "b": 1.0}},
            pares_fortemente_correlacionados=[],
        )
        resumo = AnalysisSummary(outliers=[], correlacoes=correlacao_sem_pares_fortes)

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert resultado == []


# ---------------------------------------------------------------------------
# Geração combinada
# ---------------------------------------------------------------------------


class TestGeracaoCombinada:
    """Casos em que boxplot e dispersão são gerados na mesma chamada."""

    def test_gera_boxplot_e_dispersao_simultaneamente(
        self,
        dados_numericos: pd.DataFrame,
        outlier_coluna_a: OutlierResult,
        correlacao_forte_a_b: CorrelationResult,
    ) -> None:
        resumo = AnalysisSummary(
            outliers=[outlier_coluna_a], correlacoes=correlacao_forte_a_b
        )

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        tipos_gerados = {g.tipo for g in resultado}
        assert tipos_gerados == {TipoGrafico.BOXPLOT, TipoGrafico.DISPERSAO}
        assert len(resultado) == 2


# ---------------------------------------------------------------------------
# Tipo de retorno
# ---------------------------------------------------------------------------


class TestTipoDeRetorno:
    """Garante que a API pública só devolve objetos `ChartSpec`."""

    def test_retorna_apenas_instancias_de_chartspec(
        self,
        dados_numericos: pd.DataFrame,
        outlier_coluna_a: OutlierResult,
        correlacao_forte_a_b: CorrelationResult,
    ) -> None:
        resumo = AnalysisSummary(
            outliers=[outlier_coluna_a], correlacoes=correlacao_forte_a_b
        )

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert len(resultado) > 0
        assert all(isinstance(grafico, ChartSpec) for grafico in resultado)


# ---------------------------------------------------------------------------
# Serialização da figura
# ---------------------------------------------------------------------------


class TestSerializacaoFigura:
    """Garante que `ChartSpec.figura` é sempre um dict serializável em JSON."""

    def test_figura_do_boxplot_e_serializavel_com_json_dumps(
        self, dados_numericos: pd.DataFrame, outlier_coluna_a: OutlierResult
    ) -> None:
        resumo = AnalysisSummary(outliers=[outlier_coluna_a])

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert isinstance(resultado[0].figura, dict)
        json.dumps(resultado[0].figura)  # não deve levantar TypeError

    def test_figura_da_dispersao_e_serializavel_com_json_dumps(
        self,
        dados_numericos: pd.DataFrame,
        correlacao_forte_a_b: CorrelationResult,
    ) -> None:
        resumo = AnalysisSummary(outliers=[], correlacoes=correlacao_forte_a_b)

        resultado = gerar_visualizacoes(dados_numericos, resumo)

        assert isinstance(resultado[0].figura, dict)
        json.dumps(resultado[0].figura)  # não deve levantar TypeError