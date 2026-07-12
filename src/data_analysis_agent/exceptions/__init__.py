"""Hierarquia centralizada de exceções da aplicação."""

from data_analysis_agent.exceptions.errors import (
    AnalysisError,
    ConfigurationError,
    DataAnalysisAgentError,
    DataCleaningError,
    EngineError,
    ReportGenerationError,
    VisualizationError,
)

__all__ = [
    "DataAnalysisAgentError",
    "ConfigurationError",
    "DataCleaningError",
    "AnalysisError",
    "VisualizationError",
    "ReportGenerationError",
    "EngineError",
]
