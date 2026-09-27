from __future__ import annotations
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary, CorrelationResult, OutlierResult
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ChartSpec, ReportMetadata, TipoGrafico

_APP_PATH = "../../src/data_analysis_agent/interfaces/streamlit_app/app.py"


@pytest.fixture(autouse=True)
def _limpar_cache_streamlit():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def dataframe_fake() -> pd.DataFrame:
    return pd.DataFrame({"idade": [20, 30, 40], "salario": [2000, 2500, 3000]})


@pytest.fixture
def cleaning_report_fake() -> CleaningReport:
    return CleaningReport(formato_original=(10, 3), formato_final=(9, 3), duplicatas_removidas=1, colunas_perfil=[], valores_imputados={"idade": 2}, conversoes_tipo={"salario": "object -> float64"})


@pytest.fixture
def analysis_summary_fake() -> AnalysisSummary:
    return AnalysisSummary(
        estatisticas_descritivas={"idade": {"mean": 30.0, "std": 8.16}},
        outliers=[OutlierResult(coluna="idade", metodo="iqr", limite_inferior=0.0, limite_superior=100.0, indices_outliers=[3], total_outliers=1)],
        correlacoes=CorrelationResult(metodo="pearson", matriz_correlacao={"idade": {"idade": 1.0, "salario": 0.9}}, pares_fortemente_correlacionados=[("idade", "salario", 0.9)]),
    )


@pytest.fixture
def chart_specs_fake() -> list[ChartSpec]:
    return [ChartSpec(titulo="Boxplot idade", tipo=TipoGrafico.BOXPLOT, figura={"data": [{"type": "box", "y": [20, 30, 40]}], "layout": {}}, descricao="Outlier detectado.")]


@pytest.fixture
def report_result_fake() -> ReportResult:
    return ReportResult(metadata=ReportMetadata(titulo="Relatório de teste", fonte_dados="teste.csv"), markdown="# Relatório de teste")


@pytest.fixture
def analysis_result_fake(cleaning_report_fake, analysis_summary_fake, chart_specs_fake, report_result_fake) -> AnalysisResult:
    return AnalysisResult(fonte_dados="teste.csv", relatorio_limpeza=cleaning_report_fake, resultado_analise=analysis_summary_fake, graficos=chart_specs_fake, relatorio=report_result_fake)


class TestEstadoInicial:
    def test_sem_upload_mostra_instrucao_sem_exception(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert not at.exception
        assert len(at.info) == 1
        assert "Executar análise" in at.info[0].value

    def test_sem_upload_nao_mostra_abas(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert len(at.tabs) == 0


class TestAbasComResultado:
    def test_renderiza_cinco_abas_sem_exception(self, analysis_result_fake, dataframe_fake) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["nome_arquivo"] = "teste.csv"
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()
        assert not at.exception
        assert len(at.tabs) == 5
        assert len(at.success) == 1
        assert "teste.csv" in at.success[0].value

    def test_aba_limpeza_mostra_metricas_e_perfil(self, analysis_result_fake, dataframe_fake) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()
        assert not at.exception
        assert len(at.metric) == 3

    def test_aba_analise_mostra_outliers_e_correlacoes(self, analysis_result_fake, dataframe_fake) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()
        assert not at.exception
        subheaders = [s.value for s in at.subheader]
        assert "Outliers" in subheaders
        assert "Correlações" in subheaders
        assert "Pares fortemente correlacionados" in subheaders

    def test_aba_graficos_processa_chart_spec_sem_exception(self, analysis_result_fake, dataframe_fake) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()
        assert not at.exception
        assert "Boxplot idade" in [s.value for s in at.subheader]

    def test_sem_graficos_mostra_caption_informativo(self, analysis_result_fake, dataframe_fake) -> None:
        analysis_result_fake.graficos = []
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()
        assert not at.exception
        assert any("Nenhum gráfico" in c.value for c in at.caption)


class TestAbaRelatorio:
    def test_mostra_botao_de_download_e_conteudo_markdown(
        self, analysis_result_fake, dataframe_fake
    ) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["nome_arquivo"] = "teste.csv"
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()

        assert not at.exception
        assert len(at.download_button) == 3
        assert all("Baixar relatório" in botao.label for botao in at.download_button)
        assert any(".md" in botao.label for botao in at.download_button)
        assert any(".html" in botao.label for botao in at.download_button)
        assert any(".pdf" in botao.label for botao in at.download_button)
        assert any("Relatório de teste" in m.value for m in at.markdown)

    def test_sem_relatorio_mostra_caption_informativo(
        self, analysis_result_fake, dataframe_fake
    ) -> None:
        analysis_result_fake.relatorio = None
        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()

        assert not at.exception
        assert len(at.download_button) == 0
        assert any("Nenhum relatório" in c.value for c in at.caption)


class TestAbaChat:
    def test_sem_api_key_mostra_aviso(self, analysis_result_fake, dataframe_fake, monkeypatch) -> None:
        # setenv com string vazia, não delenv: Settings lê .env como fallback,
        # e o .env real no disco tem uma key configurada — delenv sozinho
        # não isola o teste. mock.patch em cima de "from...import" não é
        # confiável com AppTest.from_file() (ver nota no app.py / SDD).
        monkeypatch.setenv("DAA_ANTHROPIC_API_KEY", "")

        at = AppTest.from_file(_APP_PATH)
        at.session_state["resultado"] = analysis_result_fake
        at.session_state["dados_brutos"] = dataframe_fake
        at.run()

        assert not at.exception
        assert len(at.warning) == 1
        assert "DAA_ANTHROPIC_API_KEY" in at.warning[0].value

    def test_com_api_key_e_pergunta_chama_agent_e_exibe_resposta(self, analysis_result_fake, dataframe_fake, monkeypatch) -> None:
        monkeypatch.setenv("DAA_ANTHROPIC_API_KEY", "fake-key")
        with patch("data_analysis_agent.agent.agent.anthropic.Anthropic"), \
             patch("data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"), \
             patch("data_analysis_agent.interfaces.streamlit_app.app.Agent.perguntar", return_value="A idade média é 30 anos.") as mock_perguntar:
            at = AppTest.from_file(_APP_PATH)
            at.session_state["resultado"] = analysis_result_fake
            at.session_state["dados_brutos"] = dataframe_fake
            at.run()
            assert not at.exception
            assert len(at.warning) == 0

            at.chat_input[0].set_value("Qual a idade média?").run()

            assert not at.exception
            mock_perguntar.assert_called_once()
            args = mock_perguntar.call_args.args
            assert args[0] == "Qual a idade média?"
            assert args[1] is dataframe_fake
            assert args[2] == "teste.csv"

            mensagens = [m.markdown[0].value for m in at.chat_message]
            assert "Qual a idade média?" in mensagens
            assert "A idade média é 30 anos." in mensagens

    def test_erro_do_agente_e_exibido_no_chat_sem_quebrar_pagina(self, analysis_result_fake, dataframe_fake, monkeypatch) -> None:
        from data_analysis_agent.exceptions.errors import EngineError
        monkeypatch.setenv("DAA_ANTHROPIC_API_KEY", "fake-key")
        with patch("data_analysis_agent.agent.agent.anthropic.Anthropic"), \
             patch("data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"), \
             patch("data_analysis_agent.interfaces.streamlit_app.app.Agent.perguntar", side_effect=EngineError("erro simulado")):
            at = AppTest.from_file(_APP_PATH)
            at.session_state["resultado"] = analysis_result_fake
            at.session_state["dados_brutos"] = dataframe_fake
            at.run()
            at.chat_input[0].set_value("Me dê insights").run()
            assert not at.exception
            mensagens = [m.markdown[0].value for m in at.chat_message]
            assert any("Erro" in m for m in mensagens)

    def test_historico_existente_e_exibido_ao_recarregar(self, analysis_result_fake, dataframe_fake, monkeypatch) -> None:
        monkeypatch.setenv("DAA_ANTHROPIC_API_KEY", "fake-key")
        with patch("data_analysis_agent.agent.agent.anthropic.Anthropic"), \
             patch("data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"):
            at = AppTest.from_file(_APP_PATH)
            at.session_state["resultado"] = analysis_result_fake
            at.session_state["dados_brutos"] = dataframe_fake
            at.session_state["historico_chat"] = [
                {"role": "user", "content": "pergunta antiga"},
                {"role": "assistant", "content": "resposta antiga"},
            ]
            at.run()
            assert not at.exception
            mensagens = [m.markdown[0].value for m in at.chat_message]
            assert "pergunta antiga" in mensagens
            assert "resposta antiga" in mensagens