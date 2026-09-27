"""Exportador de relatório em Markdown.

Conforme a Seção 7 do SDD v2.0/v2.1, a geração de Markdown não é
duplicada aqui: este exportador apenas adapta o conteúdo já produzido
por ``engine.report.gerar_relatorio_markdown`` ao contrato
``ReportExporter``, convertendo o texto para ``bytes``.
"""
from __future__ import annotations

from data_analysis_agent.engine.report import gerar_relatorio_markdown
from data_analysis_agent.models.analysis_models import AnalysisResult


class MarkdownExporter:
    """Exporta o relatório final como Markdown puro (UTF-8)."""

    extensao: str = "md"
    mime_type: str = "text/markdown"

    def exportar(self, resultado: AnalysisResult) -> bytes:
        """Gera o relatório em Markdown e o codifica em bytes.

        Args:
            resultado: resultado completo do pipeline de análise.

        Returns:
            bytes: conteúdo Markdown do relatório, codificado em UTF-8.
        """
        relatorio = gerar_relatorio_markdown(resultado)
        return relatorio.markdown.encode("utf-8")