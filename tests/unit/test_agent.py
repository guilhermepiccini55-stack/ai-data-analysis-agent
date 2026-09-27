"""Testes unitários de agent.agent.

Testes black-box da API pública (`Agent.executar_pipeline` e
`Agent.perguntar`), utilizando mocks para o client Anthropic e para a
AnalysisEngine — sem chamadas reais à API.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.config.settings import Settings
from data_analysis_agent.exceptions.errors import (
    ConfigurationError,
    DataCleaningError,
    DataAnalysisAgentError,
)
from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary, CorrelationResult, OutlierResult
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ChartSpec, ReportMetadata, TipoGrafico

_ALVO_ANTHROPIC = "data_analysis_agent.agent.agent.anthropic.Anthropic"


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="fake-key",
        anthropic_model="claude-sonnet-5",
    )


@pytest.fixture
def settings_sem_api_key():
    return Settings(anthropic_api_key=None, anthropic_model="claude-sonnet-5")


@pytest.fixture
def engine():
    return MagicMock()


@pytest.fixture
def dados():
    return pd.DataFrame({"a": [1, 2, 3]})


@pytest.fixture
def cleaning_report_fake() -> CleaningReport:
    return CleaningReport(
        formato_original=(10, 3),
        formato_final=(9, 3),
        duplicatas_removidas=1,
    )


@pytest.fixture
def analysis_summary_fake() -> AnalysisSummary:
    return AnalysisSummary(
        estatisticas_descritivas={"a": {"mean": 2.0}},
        outliers=[
            OutlierResult(
                coluna="a", metodo="iqr", limite_inferior=0.0, limite_superior=10.0, total_outliers=0
            )
        ],
        correlacoes=CorrelationResult(metodo="pearson", matriz_correlacao={"a": {"a": 1.0}}),
    )


@pytest.fixture
def chart_specs_fake() -> list[ChartSpec]:
    return [ChartSpec(titulo="Boxplot a", tipo=TipoGrafico.BOXPLOT, figura={"data": [], "layout": {}})]


@pytest.fixture
def report_result_fake() -> ReportResult:
    return ReportResult(
        metadata=ReportMetadata(titulo="Relatório", fonte_dados="teste.csv"),
        markdown="# Relatório",
    )


@pytest.fixture
def analysis_result_fake(cleaning_report_fake, analysis_summary_fake, chart_specs_fake, report_result_fake) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="teste.csv",
        relatorio_limpeza=cleaning_report_fake,
        resultado_analise=analysis_summary_fake,
        graficos=chart_specs_fake,
        relatorio=report_result_fake,
    )


def _resposta_texto(texto: str) -> SimpleNamespace:
    """Monta uma resposta simulada da API sem tool_use (fim da conversa)."""
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=texto)])


def _resposta_tool_use(nome_tool: str, tool_use_id: str = "toolu_1") -> SimpleNamespace:
    """Monta uma resposta simulada da API pedindo o uso de uma tool."""
    bloco = SimpleNamespace(type="tool_use", id=tool_use_id, name=nome_tool, input={})
    return SimpleNamespace(stop_reason="tool_use", content=[bloco])


# ---------------------------------------------------------------------------
# executar_pipeline — não exige LLM configurado
# ---------------------------------------------------------------------------


class TestExecutarPipeline:
    def test_delega_direto_para_engine(self, engine, settings_sem_api_key, dados, analysis_result_fake):
        engine.executar_pipeline.return_value = analysis_result_fake
        agent = Agent(engine, settings=settings_sem_api_key)

        resultado = agent.executar_pipeline(dados)

        assert resultado is analysis_result_fake
        engine.executar_pipeline.assert_called_once_with(dados)

    def test_nao_instancia_client_anthropic(self, engine, settings_sem_api_key, dados, analysis_result_fake):
        engine.executar_pipeline.return_value = analysis_result_fake
        agent = Agent(engine, settings=settings_sem_api_key)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            agent.executar_pipeline(dados)
            mock_anthropic.assert_not_called()


# ---------------------------------------------------------------------------
# _obter_client — instanciação lazy do client Anthropic
# ---------------------------------------------------------------------------


class TestObterClient:
    def test_perguntar_sem_api_key_levanta_configuration_error(self, engine, settings_sem_api_key, dados):
        agent = Agent(engine, settings=settings_sem_api_key)

        with pytest.raises(ConfigurationError):
            agent.perguntar("Qual a média?", dados)

    def test_client_e_criado_apenas_uma_vez_entre_chamadas(self, engine, settings, dados):
        agent = Agent(engine, settings=settings)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = _resposta_texto("ok")

            agent.perguntar("Pergunta 1", dados)
            agent.perguntar("Pergunta 2", dados)

            mock_anthropic.assert_called_once_with(api_key="fake-key")


# ---------------------------------------------------------------------------
# perguntar — fluxo de tool-calling completo
# ---------------------------------------------------------------------------


class TestPerguntar:
    def test_resposta_direta_sem_tool_use_nao_chama_engine(self, engine, settings, dados):
        agent = Agent(engine, settings=settings)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = _resposta_texto(
                "A idade média é 30 anos."
            )

            resposta = agent.perguntar("Qual a idade média?", dados)

            assert resposta == "A idade média é 30 anos."
            engine.limpar.assert_not_called()

    def test_fluxo_completo_ate_relatorio(
        self,
        engine,
        settings,
        dados,
        cleaning_report_fake,
        analysis_summary_fake,
        chart_specs_fake,
        report_result_fake,
    ):
        engine.limpar.return_value = (dados, cleaning_report_fake)
        engine.analisar.return_value = analysis_summary_fake
        engine.visualizar.return_value = chart_specs_fake
        engine.gerar_relatorio.return_value = report_result_fake

        agent = Agent(engine, settings=settings)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = [
                _resposta_tool_use("limpar_dados"),
                _resposta_tool_use("analisar_dados"),
                _resposta_tool_use("gerar_visualizacoes"),
                _resposta_tool_use("gerar_relatorio"),
                _resposta_texto("Relatório concluído."),
            ]

            resposta = agent.perguntar("Gere o relatório completo", dados)

            assert resposta == "Relatório concluído."
            engine.limpar.assert_called_once_with(dados)
            engine.analisar.assert_called_once_with(dados)
            engine.visualizar.assert_called_once_with(dados, analysis_summary_fake)
            engine.gerar_relatorio.assert_called_once()

    def test_limite_de_iteracoes_retorna_mensagem_padrao(self, engine, settings, dados):
        agent = Agent(engine, settings=settings)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = _resposta_tool_use("limpar_dados")
            engine.limpar.return_value = (dados, MagicMock())

            resposta = agent.perguntar("Pergunta sem fim", dados)

            assert "não consegui concluir" in resposta.lower()
            assert mock_anthropic.return_value.messages.create.call_count == 6

    def test_cada_chamada_a_perguntar_e_independente(
        self, engine, settings, dados, cleaning_report_fake
    ):
        """Estado intermediário de uma chamada não deve vazar para a próxima."""
        engine.limpar.return_value = (dados, cleaning_report_fake)
        agent = Agent(engine, settings=settings)

        with patch(_ALVO_ANTHROPIC) as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = [
                _resposta_tool_use("limpar_dados"),
                _resposta_texto("Primeira resposta."),
                # Segunda chamada: pede analisar_dados sem ter chamado
                # limpar_dados nesta nova chamada — deve falhar mesmo que
                # a primeira chamada já tenha limpado os dados.
                _resposta_tool_use("analisar_dados"),
                _resposta_texto("Segunda resposta."),
            ]

            agent.perguntar("Primeira pergunta", dados)
            agent.perguntar("Segunda pergunta", dados)

            engine.analisar.assert_not_called()


# ---------------------------------------------------------------------------
# _executar_tool — validação de pré-requisitos e tratamento de erros
# ---------------------------------------------------------------------------


class TestExecutarTool:
    def test_tool_desconhecida_retorna_erro(self, engine, settings):
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("tool_inexistente", {})

        assert "desconhecida" in resultado.lower()

    def test_analisar_sem_limpar_antes_retorna_erro(self, engine, settings):
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("analisar_dados", {})

        assert "erro" in resultado.lower()
        engine.analisar.assert_not_called()

    def test_visualizar_sem_prerequisitos_retorna_erro(self, engine, settings):
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("gerar_visualizacoes", {"dados_limpos": None})

        assert "erro" in resultado.lower()
        engine.visualizar.assert_not_called()

    def test_relatorio_sem_prerequisitos_retorna_erro(self, engine, settings):
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("gerar_relatorio", {"relatorio_limpeza": MagicMock()})

        assert "erro" in resultado.lower()
        engine.gerar_relatorio.assert_not_called()

    def test_insights_sem_analise_previa_retorna_erro(self, engine, settings):
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("gerar_insights", {})

        assert "erro" in resultado.lower()
        engine.gerar_insights.assert_not_called()

    def test_erro_do_engine_e_capturado_e_formatado(self, engine, settings, dados):
        engine.limpar.side_effect = DataCleaningError("dataset vazio")
        agent = Agent(engine, settings=settings)

        resultado = agent._executar_tool("limpar_dados", {"dados_brutos": dados})

        assert "dataset vazio" in resultado
        assert "limpar_dados" in resultado