"""Testes unitários de exporters.markdown.

Black-box: verifica apenas o comportamento observável do
``MarkdownExporter`` (contrato ``ReportExporter`` + bytes produzidos),
sem duplicar as asserções detalhadas de conteúdo já cobertas em
``test_report.py`` (a lógica de geração é a mesma, apenas reaproveitada).
"""
from __future__ import annotations

import pytest

from data_analysis_agent.exporters.base import ReportExporter
from data_analysis_agent.exporters.markdown import MarkdownExporter
from data_analysis_agent.models.analysis_models import (
    AnalysisResult,
    AnalysisSummary,
)
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.report_models import ChartSpec, TipoGrafico


@pytest.fixture
def cleaning_report() -> CleaningReport:
    return CleaningReport(
        formato_original=(100, 5),
        formato_final=(95, 4),
        duplicatas_removidas=3,
        valores_imputados={"idade": 2},
        conversoes_tipo={},
    )


@pytest.fixture
def analysis_summary() -> AnalysisSummary:
    return AnalysisSummary(
        estatisticas_descritivas={"idade": {"media": 30.0}},
        outliers=[],
        correlacoes=None,
    )


@pytest.fixture
def chart_specs() -> list[ChartSpec]:
    return [
        ChartSpec(titulo="Boxplot idade", tipo=TipoGrafico.BOXPLOT, figura={"data": [], "layout": {}}),
    ]


@pytest.fixture
def analysis_result(cleaning_report, analysis_summary, chart_specs) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="clientes.csv",
        relatorio_limpeza=cleaning_report,
        resultado_analise=analysis_summary,
        graficos=chart_specs,
        relatorio=None,
    )


@pytest.fixture
def analysis_result_sem_graficos(cleaning_report, analysis_summary) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="clientes.csv",
        relatorio_limpeza=cleaning_report,
        resultado_analise=analysis_summary,
        graficos=[],
        relatorio=None,
    )


class TestContrato:
    def test_satisfaz_protocol_reportexporter(self):
        exporter = MarkdownExporter()

        assert isinstance(exporter, ReportExporter)

    def test_extensao_e_mime_type(self):
        exporter = MarkdownExporter()

        assert exporter.extensao == "md"
        assert exporter.mime_type == "text/markdown"


class TestExportar:
    def test_retorna_bytes(self, analysis_result):
        conteudo = MarkdownExporter().exportar(analysis_result)

        assert isinstance(conteudo, bytes)

    def test_conteudo_corresponde_ao_gerado_por_gerar_relatorio_markdown(self, analysis_result):
        from data_analysis_agent.engine.report import gerar_relatorio_markdown

        conteudo = MarkdownExporter().exportar(analysis_result)
        esperado = gerar_relatorio_markdown(analysis_result).markdown

        assert conteudo.decode("utf-8") == esperado

    def test_funciona_sem_graficos(self, analysis_result_sem_graficos):
        conteudo = MarkdownExporter().exportar(analysis_result_sem_graficos)

        assert isinstance(conteudo, bytes)
        assert b"Total de graficos gerados: 0" in conteudo or "Total de gráficos gerados: 0".encode("utf-8") in conteudo

    def test_suporta_caracteres_unicode(self, analysis_result):
        conteudo = MarkdownExporter().exportar(analysis_result)

        assert "clientes.csv".encode("utf-8") in conteudo
        # o título usa em-dash (—), garante round-trip correto em UTF-8
        assert conteudo.decode("utf-8")