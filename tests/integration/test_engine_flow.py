import pandas as pd

from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.models.analysis_models import AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport


def test_fluxo_limpeza_e_analise() -> None:
    df = pd.DataFrame(
        {
            "idade": [20, 21, 22, 100],
            "salario": [2000, 2500, 3000, 20000],
            "nome": ["Ana", "Bruno", "Carlos", "Diana"],
        }
    )

    engine = AnalysisEngine()

    dados_limpos, relatorio_limpeza = engine.limpar(df)
    resultado_analise = engine.analisar(dados_limpos)

    assert isinstance(relatorio_limpeza, CleaningReport)
    assert isinstance(resultado_analise, AnalysisSummary)