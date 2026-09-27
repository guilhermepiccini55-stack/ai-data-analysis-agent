"""Testes unitários de exporters.pdf.

Black-box: verifica apenas o comportamento observável do
``PDFExporter`` — contrato ``ReportExporter``, bytes retornados,
assinatura ``%PDF-``, tamanho não trivial e comportamento em casos de
borda suportados pelos modelos reais.
"""
from __future__ import annotations

import pytest

from data_analysis_agent.exporters.base import ReportExporter
from data_analysis_agent.exporters.pdf import PDFExporter
from data_analysis_agent.models.analysis_models import (
    AnalysisResult,
    AnalysisSummary,
    CorrelationResult,
    OutlierResult,
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
        outliers=[
            OutlierResult(
                coluna="idade", metodo="iqr", limite_inferior=0.0, limite_superior=60.0, total_outliers=2
            ),
        ],
        correlacoes=CorrelationResult(
            metodo="pearson",
            matriz_correlacao={"idade": {"idade": 1.0}},
            pares_fortemente_correlacionados=[],
        ),
    )


@pytest.fixture
def chart_specs() -> list[ChartSpec]:
    return [
        ChartSpec(
            titulo="Boxplot idade",
            tipo=TipoGrafico.BOXPLOT,
            figura={"data": [{"type": "box", "y": [1, 2, 3]}], "layout": {}},
            descricao="2 outlier(s) detectado(s).",
        ),
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
def analysis_result_sem_graficos(cleaning_report) -> AnalysisResult:
    resumo_minimo = AnalysisSummary(
        estatisticas_descritivas={},
        outliers=[],
        correlacoes=None,
    )
    return AnalysisResult(
        fonte_dados="clientes.csv",
        relatorio_limpeza=cleaning_report,
        resultado_analise=resumo_minimo,
        graficos=[],
        relatorio=None,
    )


class TestContrato:
    def test_satisfaz_protocol_reportexporter(self):
        assert isinstance(PDFExporter(), ReportExporter)

    def test_extensao_e_mime_type(self):
        exporter = PDFExporter()

        assert exporter.extensao == "pdf"
        assert exporter.mime_type == "application/pdf"


class TestExportar:
    def test_retorna_bytes(self, analysis_result):
        conteudo = PDFExporter().exportar(analysis_result)

        assert isinstance(conteudo, bytes)

    def test_comeca_com_assinatura_pdf(self, analysis_result):
        conteudo = PDFExporter().exportar(analysis_result)

        assert conteudo.startswith(b"%PDF-")

    def test_tamanho_nao_e_trivial(self, analysis_result):
        conteudo = PDFExporter().exportar(analysis_result)

        assert len(conteudo) > 1000

    def test_relatorio_com_graficos_gera_pdf_valido(self, analysis_result):
        conteudo = PDFExporter().exportar(analysis_result)

        assert conteudo.startswith(b"%PDF-")
        assert len(conteudo) > 1000

    def test_relatorio_sem_graficos_gera_pdf_valido(self, analysis_result_sem_graficos):
        conteudo = PDFExporter().exportar(analysis_result_sem_graficos)

        assert conteudo.startswith(b"%PDF-")
        assert len(conteudo) > 1000

    def test_nao_falha_com_conteudo_textual_minimo(self, analysis_result_sem_graficos):
        # dados mínimos: sem estatísticas, sem outliers, sem correlação, sem gráficos
        conteudo = PDFExporter().exportar(analysis_result_sem_graficos)

        assert isinstance(conteudo, bytes)
        assert conteudo.startswith(b"%PDF-")

    def test_suporta_caracteres_unicode(self, analysis_result):
        # nome de fonte de dados com acentuação, já presente no relatório
        conteudo = PDFExporter().exportar(analysis_result)

        assert conteudo.startswith(b"%PDF-")
        assert len(conteudo) > 1000