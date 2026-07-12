import pandas as pd

from data_analysis_agent.engine.analysis import analisar_dados
from data_analysis_agent.models.analysis_models import AnalysisSummary


def test_analisar_dados_retorna_analysis_summary():
    dados = pd.DataFrame(
        {
            "idade": [20, 30, 40],
            "salario": [2000, 3000, 4000],
        }
    )

    resultado = analisar_dados(dados)

    assert isinstance(resultado, AnalysisSummary)

def test_analisar_dados_calcula_estatisticas_descritivas():
    dados = pd.DataFrame(
        {
            "idade": [10, 20, 30],
        }
    )

    resultado = analisar_dados(dados)

    assert "idade" in resultado.estatisticas_descritivas
    assert resultado.estatisticas_descritivas["idade"]["media"] == 20.0
    assert resultado.estatisticas_descritivas["idade"]["minimo"] == 10.0
    assert resultado.estatisticas_descritivas["idade"]["maximo"] == 30.0

def test_analisar_dados_detecta_outliers():
    dados = pd.DataFrame(
        {
            "valor": [10, 11, 12, 13, 100],
        }
    )

    resultado = analisar_dados(dados)

    assert len(resultado.outliers) == 1
    assert resultado.outliers[0].coluna == "valor"
    assert resultado.outliers[0].total_outliers == 1
    assert resultado.outliers[0].metodo.value == "iqr"

def test_analisar_dados_calcula_correlacao():
    dados = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [2, 4, 6, 8],
        }
    )

    resultado = analisar_dados(dados)

    assert resultado.correlacoes is not None
    assert resultado.correlacoes.metodo == "pearson"
    assert resultado.correlacoes.matriz_correlacao["x"]["y"] == 1.0

def test_analisar_dados_sem_colunas_numericas_retorna_vazio():
    dados = pd.DataFrame(
        {
            "nome": ["Ana", "Bruno", "Carlos"],
        }
    )

    resultado = analisar_dados(dados)

    assert resultado.estatisticas_descritivas == {}
    assert resultado.outliers == []
    assert resultado.correlacoes is None