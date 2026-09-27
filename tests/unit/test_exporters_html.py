"""Testes unitários de exporters.html.

Black-box: verifica apenas o comportamento observável do
``HTMLExporter`` — contrato ``ReportExporter``, estrutura mínima do
HTML gerado, presença do conteúdo do relatório e tratamento de
gráficos — sem inspecionar detalhes internos de implementação.
"""
from __future__ import annotations

import pytest

from data_analysis_agent.exporters.base import ReportExporter
from data_analysis_agent.exporters.html import HTMLExporter
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
        conversoes_tipo={"data_evento": "str -> datetime64[us]"},
    )


@pytest.fixture
def analysis_summary_com_correlacao() -> AnalysisSummary:
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
        ChartSpec(
            titulo="Dispersão idade x salário",
            tipo=TipoGrafico.DISPERSAO,
            figura={"data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}], "layout": {}},
        ),
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


@pytest.fixture
def analysis_result_sem_graficos(cleaning_report, analysis_summary_com_correlacao) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="clientes.csv",
        relatorio_limpeza=cleaning_report,
        resultado_analise=analysis_summary_com_correlacao,
        graficos=[],
        relatorio=None,
    )


class TestContrato:
    def test_satisfaz_protocol_reportexporter(self):
        assert isinstance(HTMLExporter(), ReportExporter)

    def test_extensao_e_mime_type(self):
        exporter = HTMLExporter()

        assert exporter.extensao == "html"
        assert exporter.mime_type == "text/html"


class TestExportar:
    def test_retorna_bytes(self, analysis_result):
        conteudo = HTMLExporter().exportar(analysis_result)

        assert isinstance(conteudo, bytes)

    def test_html_minimamente_estruturado(self, analysis_result):
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert html.strip().startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "<head" in html
        assert "<body" in html
        assert "</html>" in html

    def test_contem_conteudo_do_relatorio(self, analysis_result):
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert "Duplicatas removidas: 3" in html
        assert "clientes.csv" in html

    def test_contem_graficos_plotly_quando_presentes(self, analysis_result):
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert "Boxplot idade" in html
        assert "Dispersão idade x salário" in html
        assert "plotly" in html.lower()

    def test_html_e_autocontido_sem_recursos_externos(self, analysis_result):
        """O HTML exportado não deve depender de rede para renderizar os gráficos.

        ``include_plotlyjs="inline"`` embute o próprio código-fonte de
        ``plotly.js`` no arquivo — diferente de ``"cdn"``, que gera uma
        tag ``<script src="https://cdn.plot.ly/...">`` carregada em
        tempo de exibição. A ausência de qualquer ``<script src=`` é o
        que garante que o arquivo funciona offline.
        """
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert "<script src=" not in html

    def test_plotly_js_embutido_uma_unica_vez(self, analysis_result):
        """Mesmo com dois gráficos, o bundle de ~4.8MB do plotly.js deve
        aparecer só uma vez na página (o segundo gráfico reaproveita a
        cópia já embutida pelo primeiro). Verificado pelo tamanho total:
        uma cópia ocupa ~4.8MB; duas cópias dobrariam esse tamanho.
        """
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert 1_000_000 < len(html) < 6_000_000
        assert html.count("Plotly.newPlot") == 2

    def test_sem_graficos_nao_inclui_secao_de_graficos(self, analysis_result_sem_graficos):
        html = HTMLExporter().exportar(analysis_result_sem_graficos).decode("utf-8")

        assert "<h2>Gráficos</h2>" not in html
        assert "plotly" not in html.lower()

    def test_suporta_caracteres_unicode(self, analysis_result):
        html = HTMLExporter().exportar(analysis_result).decode("utf-8")

        assert "Relatório de Análise" in html