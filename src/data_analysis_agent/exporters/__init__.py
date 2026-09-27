"""Camada de exportação do relatório final (Fase 6, conforme SDD).

Nasce apenas nesta fase, quando existe mais de um formato real de saída
(Markdown, HTML, PDF) que justifica a abstração ``ReportExporter``.
"""
from __future__ import annotations

from data_analysis_agent.exporters.base import ReportExporter
from data_analysis_agent.exporters.html import HTMLExporter
from data_analysis_agent.exporters.markdown import MarkdownExporter
from data_analysis_agent.exporters.pdf import PDFExporter

__all__ = [
    "ReportExporter",
    "MarkdownExporter",
    "HTMLExporter",
    "PDFExporter",
]