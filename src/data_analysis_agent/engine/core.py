"""Motor de orquestração da análise de dados.

Conforme a Seção 5 do SDD v2.0, a ``AnalysisEngine`` é stateless por
design: nenhum método guarda estado de uma análise específica como
atributo de instância. Cada chamada recebe os dados de que precisa como
parâmetro e devolve um resultado completo e autossuficiente. Duas
chamadas seguidas com entradas diferentes não interferem uma na outra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_analysis_agent.engine.analysis import analisar_dados
from data_analysis_agent.engine.cleaning import limpar_dados
from data_analysis_agent.engine.report import gerar_relatorio_markdown
from data_analysis_agent.engine.visualization import gerar_visualizacoes
from data_analysis_agent.exceptions.errors import EngineError
from data_analysis_agent.llm.base import LLMProvider
from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ChartSpec


class AnalysisEngine:
    """Orquestra o pipeline de análise de dados (limpeza, análise, visualização e relatório).

    Dependências externas (ex.: provider de LLM, a partir da Fase 2) são
    recebidas por parâmetro no construtor — a engine nunca instancia suas
    próprias dependências internamente (injeção por parâmetro/construtor,
    sem framework de DI, conforme a Seção "Decisões fechadas" do SDD v2.0).
    """

    def __init__(self, **dependencias: Any) -> None:
        """Inicializa a engine recebendo dependências externas por injeção.

        Args:
            **dependencias: dependências externas que a engine venha a
                precisar (ex.: um provider de LLM na Fase 2). Nenhuma
                dependência é instanciada internamente.
        """
        self._dependencias: dict[str, Any] = dependencias

    def executar_pipeline(self, dados: pd.DataFrame | str | Path) -> AnalysisResult:
        """Executa o pipeline completo de análise sobre os dados informados.

        Args:
            dados: ``DataFrame`` já carregado ou caminho para um arquivo
                (ex.: CSV) a ser carregado.

        Returns:
            AnalysisResult: resultado completo e autossuficiente da análise,
            já incluindo o relatório final (``relatorio``) gerado ao fim
            do pipeline.
        """
        if isinstance(dados, (str, Path)):
            fonte_dados = str(dados)
            dataframe = pd.read_csv(dados)
        else:
            fonte_dados = "DataFrame fornecido diretamente"
            dataframe = dados

        dados_limpos, relatorio_limpeza = self.limpar(dataframe)
        resumo_analise = self.analisar(dados_limpos)
        graficos = self.visualizar(dados_limpos, resumo_analise)

        resultado_parcial = AnalysisResult(
            fonte_dados=fonte_dados,
            relatorio_limpeza=relatorio_limpeza,
            resultado_analise=resumo_analise,
            graficos=graficos,
            relatorio=None,
        )
        relatorio = self.gerar_relatorio(resultado_parcial)
        return resultado_parcial.model_copy(update={"relatorio": relatorio})

    def limpar(self, dados: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Executa apenas a etapa de limpeza de dados.

        Método granular pensado para uso futuro pelo Agente (Fase 3), que
        pode precisar invocar etapas isoladas do pipeline via tool-calling.

        Args:
            dados: ``DataFrame`` a ser limpo.

        Returns:
            tuple[pd.DataFrame, CleaningReport]: o ``DataFrame`` limpo e o
            relatório descrevendo as operações de limpeza realizadas.
        """
        return limpar_dados(dados)

    def analisar(self, dados: pd.DataFrame) -> AnalysisSummary:
        """Executa apenas a etapa de análise estatística.

        Args:
            dados: ``DataFrame`` (idealmente já limpo) a ser analisado.

        Returns:
            AnalysisSummary: resumo estruturado dos resultados da análise
            estatística.
        """
        return analisar_dados(dados)

    def visualizar(
        self,
        dados: pd.DataFrame,
        resumo_analise: AnalysisSummary,
    ) -> list[ChartSpec]:
        """Executa apenas a etapa de geração de visualizações.

        Args:
            dados: ``DataFrame`` a partir do qual os gráficos serão gerados.
            resumo_analise: resultado previamente produzido pela etapa
                de análise estatística.

        Returns:
            list[ChartSpec]: especificações dos gráficos gerados.
        """
        return gerar_visualizacoes(dados, resumo_analise)

    def gerar_relatorio(self, resultado: AnalysisResult) -> ReportResult:
        """Executa apenas a etapa de geração do relatório final em Markdown.

        Args:
            resultado: resultado completo de uma análise já executada
                (limpeza, análise e visualização), ainda sem o campo
                ``relatorio`` preenchido.

        Returns:
            ReportResult: metadados do relatório e seu conteúdo em Markdown.
        """
        return gerar_relatorio_markdown(resultado)

    def gerar_insights(self, resumo_analise: AnalysisSummary) -> str:
        llm_provider: LLMProvider | None = self._dependencias.get("llm_provider")
        if llm_provider is None:
            raise EngineError(
                "Nenhum provider de LLM foi injetado — instancie a "
                "AnalysisEngine com AnalysisEngine(llm_provider=...) para "
                "usar gerar_insights()."
            )
        return llm_provider.gerar_insights(resumo_analise.model_dump())