"""Endpoint de chat com tool-calling sobre um dataset (``POST /perguntas``)."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.api.dependencies import obter_agent

router = APIRouter(tags=["perguntas"])


@router.post("/perguntas")
async def perguntar(
    pergunta: str = Form(..., description="Pergunta em linguagem natural sobre o dataset."),
    arquivo: UploadFile = File(..., description="Arquivo CSV sobre o qual a pergunta será respondida."),
    agent: Agent = Depends(obter_agent),
) -> dict[str, str]:
    """Responde a uma pergunta sobre um dataset via chat com tool-calling.

    Endpoint fino: delega inteiramente a ``Agent.perguntar`` — que já
    concentra toda a lógica de tool-calling (Fase 3) e a checagem lazy
    de ``DAA_ANTHROPIC_API_KEY`` (levanta ``ConfigurationError``,
    traduzida para HTTP 503 pelo handler central em ``api/errors.py``
    quando a chave não está configurada). Nenhuma implementação
    paralela de chat/tool-calling é feita aqui.

    Cada chamada é independente — sem histórico de conversa mantido
    entre requisições, replicando a mesma decisão de escopo já adotada
    pela Fase 3 e pela interface Streamlit.

    Args:
        pergunta: pergunta em linguagem natural do usuário.
        arquivo: upload multipart do CSV sobre o qual as tools operam.
        agent: orquestrador central, injetado via ``Depends(obter_agent)``.

    Returns:
        dict[str, str]: ``{"resposta": <texto do agente>}``.

    Raises:
        HTTPException: 400 se o arquivo enviado não puder ser
            interpretado como um CSV válido.
    """
    conteudo = await arquivo.read()
    try:
        dataframe = pd.read_csv(io.BytesIO(conteudo))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Não foi possível ler o arquivo CSV enviado: {exc}",
        ) from exc

    fonte_dados = arquivo.filename or "DataFrame fornecido diretamente"
    resposta = agent.perguntar(pergunta, dataframe, fonte_dados=fonte_dados)
    return {"resposta": resposta}
