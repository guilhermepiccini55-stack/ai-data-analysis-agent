"""Exportador de relatório em PDF.

Conforme a Seção 9 do prompt de Fase 6: reutiliza integralmente o HTML
já produzido por ``HTMLExporter`` (que por sua vez reutiliza o Markdown
existente — nenhuma lógica de conteúdo é duplicada aqui) e o converte
para PDF com ``xhtml2pdf``.

Validado empiricamente (não apenas assumido) que ``xhtml2pdf`` ignora
com segurança o ``<script>`` embutido pelo ``HTMLExporter`` (o bundle
inline de ``plotly.js`` e os gráficos em si): o conteúdo textual do
relatório (títulos, listas, descrições dos gráficos) é preservado no
PDF, sem lixo proveniente do script. Isso satisfaz a decisão já fechada
na Seção 5.3: o PDF é uma representação estática do conteúdo textual
disponível no HTML, sem imagens dos gráficos e sem ``kaleido``.
"""
from __future__ import annotations

import io

from xhtml2pdf import pisa

from data_analysis_agent.exceptions.errors import ReportGenerationError
from data_analysis_agent.exporters.html import HTMLExporter
from data_analysis_agent.models.analysis_models import AnalysisResult


class PDFExporter:
    """Exporta o relatório final como PDF, a partir do HTML já existente."""

    extensao: str = "pdf"
    mime_type: str = "application/pdf"

    def __init__(self) -> None:
        self._html_exporter = HTMLExporter()

    def exportar(self, resultado: AnalysisResult) -> bytes:
        """Gera o relatório em PDF a partir do HTML produzido por ``HTMLExporter``.

        Args:
            resultado: resultado completo do pipeline de análise.

        Returns:
            bytes: conteúdo do PDF gerado.

        Raises:
            ReportGenerationError: se ``xhtml2pdf`` reportar erro(s) ao
                converter o HTML para PDF.
        """
        html_bytes = self._html_exporter.exportar(resultado)

        buffer = io.BytesIO()
        status = pisa.CreatePDF(
            io.BytesIO(html_bytes),
            dest=buffer,
            encoding="utf-8",
        )
        if status.err:
            raise ReportGenerationError(
                f"xhtml2pdf reportou {status.err} erro(s) ao converter o "
                "relatório HTML para PDF."
            )
        return buffer.getvalue()