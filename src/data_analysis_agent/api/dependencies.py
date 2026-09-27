"""Injeção de dependências da API REST.

Monta o :class:`~data_analysis_agent.agent.agent.Agent` da mesma forma
que a interface Streamlit o faz em
``interfaces/streamlit_app/app.py::_obter_agent`` — sem framework de DI
(conforme a Seção 5 do SDD v2.0: "injeção por parâmetro/construtor
simples"), apenas uma função de fábrica cacheada, usada como dependência
do FastAPI via ``Depends(obter_agent)``.
"""
from __future__ import annotations

from functools import lru_cache

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.config.settings import obter_settings
from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.llm.anthropic_provider import AnthropicProvider


@lru_cache(maxsize=1)
def obter_agent() -> Agent:
    """Cria (uma única vez por processo) o ``Agent`` usado pelos routers.

    Réplica da lógica usada por ``streamlit_app/app.py::_obter_agent``:
    injeta um ``AnthropicProvider`` na ``AnalysisEngine`` somente se
    ``DAA_ANTHROPIC_API_KEY`` estiver configurada. Isso permite que
    ``POST /analises`` funcione mesmo sem LLM configurado, enquanto
    ``POST /perguntas`` falha de forma explícita e controlada
    (``ConfigurationError``, traduzida para HTTP 503 pelo handler
    central) quando a chave não está presente.

    A própria ``AnalysisEngine``/``Agent`` continuam stateless — o que é
    cacheado aqui é apenas a instância do orquestrador (equivalente ao
    client HTTP/DB de qualquer aplicação FastAPI), nunca o estado de uma
    análise específica.

    Returns:
        Agent: instância pronta para uso pelos endpoints via
        ``Depends(obter_agent)``. Em testes, é substituída via
        ``app.dependency_overrides[obter_agent]``, sem instanciar
        nenhum client Anthropic real.
    """
    settings = obter_settings()
    llm_provider = AnthropicProvider(settings=settings) if settings.anthropic_api_key else None
    engine = AnalysisEngine(llm_provider=llm_provider) if llm_provider else AnalysisEngine()
    return Agent(engine, settings=settings)
