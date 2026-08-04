"""Modelos de domínio relacionados ao resultado agregado do relatório final.

Apenas estrutura — nenhuma lógica de geração de relatório vive aqui; essa
responsabilidade é exclusiva de ``engine/report.py``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from data_analysis_agent.models.report_models import ReportMetadata


class ReportResult(BaseModel):
    """Agrega os metadados do relatório final e seu conteúdo em Markdown.

    Desempenha, para a etapa de geração de relatório, o mesmo papel que
    ``CleaningReport`` desempenha para a limpeza e ``AnalysisSummary``
    para a análise estatística: um modelo estreito dedicado à saída de
    uma única etapa do pipeline.
    """

    metadata: ReportMetadata = Field(..., description="Metadados do relatório gerado.")
    markdown: str = Field(..., description="Conteúdo do relatório em formato Markdown.")