"""Tratamento centralizado das exceções da aplicação na camada HTTP.

Único lugar onde a hierarquia de :class:`DataAnalysisAgentError` é
traduzida para respostas HTTP. Os routers nunca capturam essas exceções
individualmente — apenas deixam propagar — mantendo os endpoints finos
(Seção 6 do SDD v2.0: cada adaptador cuida só da sua própria camada, sem
lógica de negócio).

Mapeamento adotado:

- ``ConfigurationError`` / ``LLMProviderError`` -> 503: o servidor não
  está pronto para atender a requisição (ex.: LLM não configurado),
  independentemente do que o cliente enviou.
- ``DataCleaningError``, ``AnalysisError``, ``VisualizationError``,
  ``ReportGenerationError``, ``EngineError`` -> 422: falha de negócio
  decorrente dos dados fornecidos pelo cliente.
- Qualquer outra ``DataAnalysisAgentError`` não categorizada acima -> 500
  (fallback defensivo, cobre futuras subclasses).
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from data_analysis_agent.exceptions.errors import (
    AnalysisError,
    ConfigurationError,
    DataAnalysisAgentError,
    DataCleaningError,
    EngineError,
    LLMProviderError,
    ReportGenerationError,
    VisualizationError,
)

_ERROS_DADO_INVALIDO: tuple[type[DataAnalysisAgentError], ...] = (
    DataCleaningError,
    AnalysisError,
    VisualizationError,
    ReportGenerationError,
    EngineError,
)

_ERROS_INDISPONIVEL: tuple[type[DataAnalysisAgentError], ...] = (
    ConfigurationError,
    LLMProviderError,
)


def _resposta_erro(status_code: int, exc: DataAnalysisAgentError) -> JSONResponse:
    """Monta o corpo JSON padrão de erro da API.

    Args:
        status_code: código HTTP a ser devolvido.
        exc: exceção de domínio capturada.

    Returns:
        JSONResponse: ``{"erro": <NomeDaClasse>, "mensagem": <str>}``.
    """
    return JSONResponse(
        status_code=status_code,
        content={"erro": type(exc).__name__, "mensagem": exc.mensagem},
    )


async def _handler_dado_invalido(request: Request, exc: DataAnalysisAgentError) -> JSONResponse:
    return _resposta_erro(status.HTTP_422_UNPROCESSABLE_CONTENT, exc)


async def _handler_indisponivel(request: Request, exc: DataAnalysisAgentError) -> JSONResponse:
    return _resposta_erro(status.HTTP_503_SERVICE_UNAVAILABLE, exc)


async def _handler_generico(request: Request, exc: DataAnalysisAgentError) -> JSONResponse:
    return _resposta_erro(status.HTTP_500_INTERNAL_SERVER_ERROR, exc)


def registrar_exception_handlers(app: FastAPI) -> None:
    """Registra, no ``app`` FastAPI, os handlers para toda a hierarquia de erros.

    Chamada uma única vez, na criação da aplicação (``api/app.py``). O
    Starlette resolve o handler pela classe mais específica na MRO da
    exceção levantada, então registrar tanto as subclasses quanto a base
    (``DataAnalysisAgentError``) é seguro: uma ``DataCleaningError``
    sempre usa o handler 422, mesmo com o fallback 500 registrado para a
    base.

    Args:
        app: instância da aplicação FastAPI.
    """
    for excecao in _ERROS_DADO_INVALIDO:
        app.add_exception_handler(excecao, _handler_dado_invalido)
    for excecao in _ERROS_INDISPONIVEL:
        app.add_exception_handler(excecao, _handler_indisponivel)
    app.add_exception_handler(DataAnalysisAgentError, _handler_generico)
