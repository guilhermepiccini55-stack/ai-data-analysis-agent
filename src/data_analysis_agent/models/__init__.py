"""Modelos de domínio (Pydantic) do AI Data Analysis Agent.

Apenas estrutura de dados — nenhum modelo aqui contém lógica de negócio.
"""

from data_analysis_agent.models.analysis_models import (
    AnalysisResult,
    CorrelationResult,
    MetodoOutlier,
    OutlierResult,
)
from data_analysis_agent.models.data_models import CleaningReport, ColumnProfile
from data_analysis_agent.models.report_models import ChartSpec, ReportMetadata, TipoGrafico

__all__ = [
    "ColumnProfile",
    "CleaningReport",
    "MetodoOutlier",
    "OutlierResult",
    "CorrelationResult",
    "AnalysisResult",
    "TipoGrafico",
    "ChartSpec",
    "ReportMetadata",
]
