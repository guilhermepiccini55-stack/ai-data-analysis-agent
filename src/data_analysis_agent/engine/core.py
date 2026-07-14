"""Motor de orquestração da análise de dados.

Conforme a Seção 5 do SDD v2.0, a ``AnalysisEngine`` é stateless por
design: nenhum método guarda estado de uma análise específica como
atributo de instância. Cada chamada recebe os dados de que precisa como
parâmetro e devolve um resultado completo e autossuficiente. Duas
chamadas seguidas com entradas diferentes não interferem uma na outra.

Nenhuma lógica de negócio está implementada nesta fase — todos os métodos
expõem apenas a interface pública definida no SDD, levantando
``NotImplementedError`` até que a implementação real seja feita.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_analysis_agent.engine.visualization import gerar_visualizacoes
from data_analysis_agent.engine.analysis import analisar_dados
from data_analysis_agent.engine.cleaning import limpar_dados
from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport
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
            AnalysisResult: resultado completo e autossuficiente da análise.

        Raises:
            NotImplementedError: a implementação do pipeline ainda não
                existe nesta fase do projeto.
        """
        raise NotImplementedError(
            "executar_pipeline() ainda não foi implementado — "
            "fundação da Fase 1 não inclui lógica de negócio."
        )

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

        Raises:
            NotImplementedError: a implementação da análise ainda não existe.
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

    def gerar_relatorio(self, resultado: AnalysisResult) -> str:
        """Executa apenas a etapa de geração do relatório final em Markdown.

        Args:
            resultado: resultado completo de uma análise já executada.

        Returns:
            str: conteúdo do relatório em formato Markdown.

        Raises:
            NotImplementedError: a implementação da geração de relatório ainda não existe.
        """
        raise NotImplementedError(
            "gerar_relatorio() ainda não foi implementado — "
            "fundação da Fase 1 não inclui lógica de negócio."
        )