"""Etapa de análise estatística do pipeline de dados.

Conforme a Seção 5 do SDD v2.0, este módulo é stateless: a única função
pública, ``analisar_dados``, recebe um ``DataFrame`` e devolve um
``AnalysisSummary`` completo, sem depender de nenhum estado externo ao
próprio dataset recebido.

A API pública deste módulo é intencionalmente mínima — apenas
``analisar_dados``. Toda a lógica de detecção de outliers, cálculo de
correlação e cálculo de estatísticas descritivas é privada, por não haver,
nesta fase, nenhum consumidor real que precise invocar essas estratégias
isoladamente (mesmo princípio de escopo incremental já aplicado em
``cleaning.py``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from data_analysis_agent.exceptions.errors import AnalysisError
from data_analysis_agent.models.analysis_models import (
    AnalysisSummary,
    CorrelationResult,
    MetodoOutlier,
    OutlierResult,
)

# Limiar (em módulo) acima do qual um par de colunas é considerado
# fortemente correlacionado. Valor de segurança inicial, análogo ao papel
# do limiar de conversão de tipo em ``cleaning.py``.
_LIMIAR_CORRELACAO_FORTE = 0.7

# Método de correlação suportado nesta primeira versão do módulo.
_METODO_CORRELACAO = "pearson"


def analisar_dados(dados: pd.DataFrame) -> AnalysisSummary:
    """Executa a análise estatística completa sobre um DataFrame.

    Calcula estatísticas descritivas, detecta outliers (método IQR) e
    calcula correlações (método Pearson) exclusivamente sobre as colunas
    numéricas do dataset. Datasets sem colunas numéricas, ou com menos de
    duas colunas numéricas (no caso da correlação), produzem um
    ``AnalysisSummary`` com os campos correspondentes vazios/``None``, sem
    levantar exceção.

    Args:
        dados: ``DataFrame`` a ser analisado (idealmente já limpo).

    Returns:
        AnalysisSummary: resumo estruturado com estatísticas descritivas,
        outliers detectados e resultado de correlação, quando aplicável.

    Raises:
        AnalysisError: se ``dados`` não for uma instância de ``pd.DataFrame``.
    """
    _validar_tipo_dataframe(dados)

    estatisticas_descritivas = _calcular_estatisticas(dados)
    outliers = _detectar_outliers(dados)
    correlacoes = _calcular_correlacoes(dados)

    return AnalysisSummary(
        estatisticas_descritivas=estatisticas_descritivas,
        outliers=outliers,
        correlacoes=correlacoes,
    )


def _validar_tipo_dataframe(dados: pd.DataFrame) -> None:
    """Garante que ``dados`` é um ``pd.DataFrame``, levantando ``AnalysisError`` caso contrário."""
    if not isinstance(dados, pd.DataFrame):
        raise AnalysisError(
            f"Tipo de entrada inválido para análise: esperado pd.DataFrame, "
            f"recebido {type(dados).__name__}."
        )


def _colunas_numericas(dados: pd.DataFrame) -> pd.Index:
    """Devolve o subconjunto de colunas de ``dados`` com dtype numérico."""
    return dados.select_dtypes(include="number").columns


def _calcular_estatisticas(dados: pd.DataFrame) -> dict[str, Any]:
    """Calcula estatísticas descritivas por coluna numérica.

    Colunas sem nenhum valor não-nulo são ignoradas (não há estatística
    calculável). Todos os valores são convertidos para tipos nativos do
    Python, mantendo o dicionário resultante serializável.
    """
    estatisticas: dict[str, Any] = {}

    for coluna in _colunas_numericas(dados):
        serie = dados[coluna].dropna()
        if serie.empty:
            continue

        desvio_padrao = serie.std()
        estatisticas[coluna] = {
            "media": float(serie.mean()),
            "mediana": float(serie.median()),
            "desvio_padrao": float(desvio_padrao) if pd.notna(desvio_padrao) else 0.0,
            "minimo": float(serie.min()),
            "maximo": float(serie.max()),
            "q1": float(serie.quantile(0.25)),
            "q3": float(serie.quantile(0.75)),
        }

    return estatisticas


def _detectar_outliers(dados: pd.DataFrame) -> list[OutlierResult]:
    """Detecta outliers em cada coluna numérica pelo método IQR.

    Colunas com variância zero (IQR == 0) são ignoradas, pois nesse caso
    qualquer valor distinto da mediana seria erroneamente sinalizado como
    outlier. Colunas sem valores válidos suficientes para calcular
    quartis também são ignoradas.
    """
    resultados: list[OutlierResult] = []

    for coluna in _colunas_numericas(dados):
        serie = dados[coluna].dropna()
        if serie.empty:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        mascara_outliers = (serie < limite_inferior) | (serie > limite_superior)
        indices_outliers = serie.index[mascara_outliers].tolist()

        resultados.append(
            OutlierResult(
                coluna=coluna,
                metodo=MetodoOutlier.IQR,
                limite_inferior=float(limite_inferior),
                limite_superior=float(limite_superior),
                indices_outliers=indices_outliers,
                total_outliers=len(indices_outliers),
            )
        )

    return resultados


def _calcular_correlacoes(dados: pd.DataFrame) -> CorrelationResult | None:
    """Calcula a matriz de correlação (Pearson) entre colunas numéricas.

    Colunas com variância zero são excluídas do cálculo, pois produzem
    correlação indefinida (NaN) com qualquer outra coluna. Devolve
    ``None`` quando restarem menos de duas colunas elegíveis.
    """
    colunas_numericas = _colunas_numericas(dados)
    if len(colunas_numericas) < 2:
        return None

    subconjunto = dados[colunas_numericas]
    colunas_com_variancia = [
        coluna for coluna in colunas_numericas if subconjunto[coluna].std() not in (0, None) and pd.notna(subconjunto[coluna].std())
    ]

    if len(colunas_com_variancia) < 2:
        return None

    matriz = subconjunto[colunas_com_variancia].corr(method=_METODO_CORRELACAO)

    matriz_correlacao: dict[str, dict[str, float]] = {
        coluna_a: {
            coluna_b: float(matriz.loc[coluna_a, coluna_b]) for coluna_b in matriz.columns
        }
        for coluna_a in matriz.index
    }

    pares_fortemente_correlacionados: list[tuple[str, str, float]] = []
    colunas_ordenadas = list(matriz.columns)
    for i, coluna_a in enumerate(colunas_ordenadas):
        for coluna_b in colunas_ordenadas[i + 1 :]:
            valor = matriz.loc[coluna_a, coluna_b]
            if pd.notna(valor) and abs(valor) >= _LIMIAR_CORRELACAO_FORTE:
                pares_fortemente_correlacionados.append((coluna_a, coluna_b, float(valor)))

    return CorrelationResult(
        metodo=_METODO_CORRELACAO,
        matriz_correlacao=matriz_correlacao,
        pares_fortemente_correlacionados=pares_fortemente_correlacionados,
    )