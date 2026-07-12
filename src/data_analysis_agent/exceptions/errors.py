"""Hierarquia centralizada de exceções da aplicação.

Antecipada da Fase 4 para a Fase 1, conforme a Seção 8 do SDD v2.0. Cada
camada (engine, interfaces, futura API) deve levantar a exceção específica
correspondente à etapa em que o erro ocorreu, sempre derivada de
:class:`DataAnalysisAgentError`.
"""
from __future__ import annotations


class DataAnalysisAgentError(Exception):
    """Exceção base de todos os erros previstos da aplicação."""

    def __init__(self, mensagem: str) -> None:
        """Inicializa a exceção com uma mensagem descritiva.

        Args:
            mensagem: descrição legível do erro ocorrido.
        """
        self.mensagem = mensagem
        super().__init__(mensagem)


class ConfigurationError(DataAnalysisAgentError):
    """Erro relacionado a configurações inválidas ou ausentes da aplicação."""


class DataCleaningError(DataAnalysisAgentError):
    """Erro ocorrido durante a etapa de limpeza de dados."""


class AnalysisError(DataAnalysisAgentError):
    """Erro ocorrido durante a etapa de análise estatística."""


class VisualizationError(DataAnalysisAgentError):
    """Erro ocorrido durante a geração de visualizações."""


class ReportGenerationError(DataAnalysisAgentError):
    """Erro ocorrido durante a geração do relatório final."""


class EngineError(DataAnalysisAgentError):
    """Erro genérico de orquestração ocorrido na AnalysisEngine."""
