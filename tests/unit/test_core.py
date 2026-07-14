import pandas as pd
import pytest

from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.models.analysis_models import AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.report_models import ChartSpec


def test_analisar_retorna_analysis_summary() -> None:
    df = pd.DataFrame(
        {
            "idade": [20, 21, 22],
            "salario": [2000, 2500, 3000],
        }
    )

    engine = AnalysisEngine()

    resultado = engine.analisar(df)

    assert isinstance(resultado, AnalysisSummary)


def test_limpar_retorna_dataframe_e_cleaning_report() -> None:
    df = pd.DataFrame(
        {
            "nome": ["Ana", "Bruno"],
            "idade": [20, 21],
        }
    )

    engine = AnalysisEngine()

    dados_limpos, relatorio = engine.limpar(df)

    assert isinstance(dados_limpos, pd.DataFrame)
    assert isinstance(relatorio, CleaningReport)


def test_visualizar_delega_para_gerar_visualizacoes():
    dados = pd.DataFrame({"idade": [20, 30, 40]})
    resumo = AnalysisSummary()

    engine = AnalysisEngine()

    resultado = engine.visualizar(dados, resumo)

    assert isinstance(resultado, list)

def test_gerar_relatorio_ainda_nao_implementado() -> None:
    engine = AnalysisEngine()

    with pytest.raises(NotImplementedError):
        engine.gerar_relatorio(None)  # type: ignore[arg-type]