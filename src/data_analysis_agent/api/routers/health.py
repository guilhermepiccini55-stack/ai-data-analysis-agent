"""Endpoint de verificação de disponibilidade da API."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Verificação simples de disponibilidade do processo da API.

    Não toca a ``AnalysisEngine`` nem o ``Agent`` — confirma apenas que
    a API está de pé e respondendo, sem depender de configuração
    externa (ex.: chave da Anthropic).

    Returns:
        dict[str, str]: ``{"status": "ok"}``.
    """
    return {"status": "ok"}
