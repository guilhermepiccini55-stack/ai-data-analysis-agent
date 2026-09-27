"""Endpoint de execução do pipeline completo de análise (``POST /analises``)."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.api.dependencies import obter_agent
from data_analysis_agent.models.analysis_models import AnalysisResult

router = APIRouter(tags=["análises"])


@router.post("/analises", response_model=AnalysisResult)
async def executar_analise(
    arquivo: UploadFile = File(..., description="Arquivo CSV a ser analisado."),
    agent: Agent = Depends(obter_agent),
) -> AnalysisResult:
    """Executa o pipeline completo de análise sobre um arquivo CSV enviado.

    Endpoint fino: apenas decodifica o upload em um ``DataFrame`` e
    delega toda a lógica de negócio ao ``Agent`` (que por sua vez
    delega à ``AnalysisEngine``, stateless, conforme Seção 5 do SDD
    v2.0). Nenhum estado é mantido entre requisições — cada chamada
    envia o dataset e recebe de volta o ``AnalysisResult`` completo e
    autossuficiente, para a interface decidir se e como cacheá-lo.

    Args:
        arquivo: upload multipart do CSV a ser analisado.
        agent: orquestrador central, injetado via ``Depends(obter_agent)``.

    Returns:
        AnalysisResult: resultado completo do pipeline (limpeza, análise
        estatística, gráficos e relatório).

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

    return agent.executar_pipeline(dataframe)
