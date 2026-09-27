"""Ponto de entrada da API REST (FastAPI) do AI Data Analysis Agent.

Fase 5 do SDD v2.0: a API é um adaptador fino sobre o ``Agent``
(orquestrador central definido na Fase 4), assim como a interface
Streamlit. Não existe ``api/schemas.py`` — os endpoints reusam os
modelos de domínio (Pydantic) de ``models/`` diretamente (Seção 6 do
SDD v2.0).

Uso local:
    uvicorn data_analysis_agent.api.app:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from data_analysis_agent.api.errors import registrar_exception_handlers
from data_analysis_agent.api.routers import analises, health, perguntas
from data_analysis_agent.config.settings import obter_settings

_settings = obter_settings()

app = FastAPI(
    title=_settings.app_name,
    description=(
        "API REST do AI Data Analysis Agent — pipeline de análise de "
        "dados e chat com tool-calling, via Agent (Fase 4)."
    ),
    version="0.1.0",
)

registrar_exception_handlers(app)

app.include_router(health.router)
app.include_router(analises.router)
app.include_router(perguntas.router)
