"""Testes unitários de engine.report.

Testes black-box: verificam o `ReportResult` (metadados + Markdown)
produzido por `gerar_relatorio_markdown` a partir de um `AnalysisResult`
completo, checando presença/contagens no texto gerado em vez de
comparar o Markdown inteiro (frágil a mudanças de formatação).
"""
from __future__ import annotations

import pytest

from data_analysis_agent.engine.report import gerar_relatorio_markdown
from data_analysis_agent.models.analysis_models import (
    AnalysisResult,
    AnalysisSummary,
    CorrelationResult,
    OutlierResult,
)
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ChartSpec, TipoGrafico


@pytest.fixture
def cleaning_report() -> CleaningReport:
    return CleaningReport(
        formato_original=(100, 5),
        formato_final=(95, 4),
        duplicatas_removidas=3,
        valores_imputados={"idade": 2, "salario": 1},
        conversoes_tipo={"data_evento": "str -> datetime64[us]"},
    )


@pytest.fixture
def analysis_summary_com_correlacao() -> AnalysisSummary:
    return AnalysisSummary(
        estatisticas_descritivas={"idade": {"media": 30.0}, "salario": {"media": 3000.0}},
        outliers=[
            OutlierResult(
                coluna="idade", metodo="iqr", limite_inferior=0.0, limite_superior=60.0, total_outliers=2
            ),
            OutlierResult(
                coluna="salario", metodo="iqr", limite_inferior=0.0, limite_superior=9000.0, total_outliers=1
            ),
        ],
        correlacoes=CorrelationResult(
            metodo="pearson",
            matriz_correlacao={"idade": {"idade": 1.0, "salario": 0.8}, "salario": {"idade": 0.8, "salario": 1.0}},
            pares_fortemente_correlacionados=[("idade", "salario", 0.8)],
        ),
    )


@pytest.fixture
def chart_specs() -> list[ChartSpec]:
    return [
        ChartSpec(titulo="Boxplot idade", tipo=TipoGrafico.BOXPLOT, figura={"data": [], "layout": {}}),
        ChartSpec(titulo="Dispersão idade x salário", tipo=TipoGrafico.DISPERSAO, figura={"data": [], "layout": {}}),
    ]


@pytest.fixture
def analysis_result(cleaning_report, analysis_summary_com_correlacao, chart_specs) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="clientes.csv",
        relatorio_limpeza=cleaning_report,
        resultado_analise=analysis_summary_com_correlacao,
        graficos=chart_specs,
        relatorio=None,
    )


class TestMetadata:
    def test_titulo_inclui_fonte_dados(self, analysis_result):
        resultado = gerar_relatorio_markdown(analysis_result)

        assert resultado.metadata.titulo == "Relatório de Análise — clientes.csv"

    def test_fonte_dados_e_autor(self, analysis_result):
        resultado = gerar_relatorio_markdown(analysis_result)

        assert resultado.metadata.fonte_dados == "clientes.csv"
        assert resultado.metadata.autor == "AI Data Analysis Agent"

    def test_secoes_fixas(self, analysis_result):
        resultado = gerar_relatorio_markdown(analysis_result)

        assert resultado.metadata.secoes == ["Limpeza de Dados", "Análise Estatística", "Visualizações"]

    def test_retorna_report_result(self, analysis_result):
        resultado = gerar_relatorio_markdown(analysis_result)

        assert isinstance(resultado, ReportResult)
        assert resultado.markdown  # não vazio


class TestSecaoLimpeza:
    def test_contem_formato_original_e_final(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "100 linhas × 5 colunas" in markdown
        assert "95 linhas × 4 colunas" in markdown

    def test_contem_duplicatas_removidas(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Duplicatas removidas: 3" in markdown

    def test_conta_colunas_imputadas_e_convertidas(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Colunas com valores imputados: 2" in markdown
        assert "Colunas com conversão de tipo: 1" in markdown


class TestSecaoAnalise:
    def test_contem_contagem_estatisticas_e_outliers(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Colunas com estatísticas descritivas: 2" in markdown
        assert "Colunas analisadas para outliers: 2" in markdown

    def test_total_outliers_soma_todas_as_colunas(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Total de outliers identificados: 3" in markdown

    def test_pares_correlacionados_quando_presente(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Pares fortemente correlacionados: 1" in markdown

    def test_correlacao_ausente_quando_none(self, cleaning_report, chart_specs):
        resumo_sem_correlacao = AnalysisSummary(
            estatisticas_descritivas={"idade": {"media": 30.0}},
            outliers=[],
            correlacoes=None,
        )
        resultado_sem_correlacao = AnalysisResult(
            fonte_dados="clientes.csv",
            relatorio_limpeza=cleaning_report,
            resultado_analise=resumo_sem_correlacao,
            graficos=chart_specs,
        )

        markdown = gerar_relatorio_markdown(resultado_sem_correlacao).markdown

        assert "Correlação: não aplicável (dados insuficientes)" in markdown
        assert "Pares fortemente correlacionados" not in markdown


class TestSecaoVisualizacoes:
    def test_lista_titulos_e_tipos_dos_graficos(self, analysis_result):
        markdown = gerar_relatorio_markdown(analysis_result).markdown

        assert "Total de gráficos gerados: 2" in markdown
        assert "Boxplot idade (boxplot)" in markdown
        assert "Dispersão idade x salário (scatter)" in markdown

    def test_total_zero_quando_sem_graficos(self, cleaning_report, analysis_summary_com_correlacao):
        resultado_sem_graficos = AnalysisResult(
            fonte_dados="clientes.csv",
            relatorio_limpeza=cleaning_report,
            resultado_analise=analysis_summary_com_correlacao,
            graficos=[],
        )

        markdown = gerar_relatorio_markdown(resultado_sem_graficos).markdown

        assert "Total de gráficos gerados: 0" in markdown


class TestIdempotenciaEIndependencia:
    def test_relatorio_ja_preenchido_no_resultado_e_ignorado_e_regerado(
        self, cleaning_report, analysis_summary_com_correlacao, chart_specs
    ):
        relatorio_antigo = ReportResult(
            metadata={"titulo": "Antigo", "fonte_dados": "antigo.csv", "secoes": []},
            markdown="# Antigo",
        )
        resultado_com_relatorio_previo = AnalysisResult(
            fonte_dados="clientes.csv",
            relatorio_limpeza=cleaning_report,
            resultado_analise=analysis_summary_com_correlacao,
            graficos=chart_specs,
            relatorio=relatorio_antigo,
        )

        novo_relatorio = gerar_relatorio_markdown(resultado_com_relatorio_previo)

        assert novo_relatorio.metadata.fonte_dados == "clientes.csv"
        assert "# Antigo" not in novo_relatorio.markdown