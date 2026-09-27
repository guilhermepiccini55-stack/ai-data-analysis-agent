"""Testes unitários da interface Streamlit.

Os testes verificam a integração da interface com o Agent, sem executar
uma aplicação Streamlit real e sem realizar chamadas à API da Anthropic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

from data_analysis_agent.agent.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _limpar_cache_streamlit():
    """Limpa o cache de `@st.cache_resource` antes/depois de cada teste.

    Necessário porque `_obter_agent()` em app.py é decorado com
    `@st.cache_resource`: sem isso, o resultado (e os mocks internos que
    o criaram) vazam de um teste para o próximo dentro do mesmo processo
    pytest.
    """
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def dados() -> pd.DataFrame:
    """DataFrame simples utilizado nos testes."""
    return pd.DataFrame(
        {
            "idade": [20, 25, 30],
            "salario": [2000, 3000, 4000],
        }
    )


@pytest.fixture
def agent_mock() -> MagicMock:
    """Mock do Agent utilizado pela interface."""
    return MagicMock(spec=Agent)


@pytest.fixture
def resultado_mock() -> MagicMock:
    """Resultado fictício produzido pelo pipeline."""
    resultado = MagicMock()
    resultado.fonte_dados = "dados.csv"
    resultado.relatorio_limpeza = MagicMock()
    resultado.resultado_analise = MagicMock()
    resultado.graficos = []
    resultado.relatorio = None
    return resultado


# ---------------------------------------------------------------------------
# Importação do módulo
# ---------------------------------------------------------------------------


class TestImportacaoApp:
    """Verifica se o módulo da interface pode ser importado."""

    def test_app_pode_ser_importado(self) -> None:
        """O módulo app deve ser importável."""
        import data_analysis_agent.interfaces.streamlit_app.app as app

        assert app is not None


# ---------------------------------------------------------------------------
# _obter_agent
# ---------------------------------------------------------------------------


class TestObterAgent:
    """Testes da criação do Agent central da aplicação."""

    def test_obter_agent_retorna_agent(self) -> None:
        """_obter_agent deve retornar uma instância de Agent."""
        from data_analysis_agent.interfaces.streamlit_app.app import _obter_agent

        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.Agent",
            return_value=agent_mock,
        ):
            resultado = _obter_agent()

        assert resultado is agent_mock

    def test_obter_agent_cria_engine_sem_provider_quando_nao_ha_api_key(
        self,
    ) -> None:
        """Sem API key, a interface deve criar a engine normalmente."""
        from data_analysis_agent.interfaces.streamlit_app.app import _obter_agent

        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = None

        engine_mock = MagicMock()
        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.AnalysisEngine",
            return_value=engine_mock,
        ) as mock_engine, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.Agent",
            return_value=agent_mock,
        ):
            resultado = _obter_agent()

        mock_engine.assert_called_once_with()
        assert resultado is agent_mock

    def test_obter_agent_cria_provider_quando_ha_api_key(
        self,
    ) -> None:
        """Com API key, a interface deve configurar o provider Anthropic."""
        from data_analysis_agent.interfaces.streamlit_app.app import _obter_agent

        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = "chave-teste"

        provider_mock = MagicMock()
        engine_mock = MagicMock()
        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.AnthropicProvider",
            return_value=provider_mock,
        ) as mock_provider, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.AnalysisEngine",
            return_value=engine_mock,
        ) as mock_engine, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.Agent",
            return_value=agent_mock,
        ):
            resultado = _obter_agent()

        mock_provider.assert_called_once_with(settings=settings_mock)
        mock_engine.assert_called_once_with(llm_provider=provider_mock)
        assert resultado is agent_mock


# ---------------------------------------------------------------------------
# Integração Agent → pipeline
# ---------------------------------------------------------------------------


class TestExecucaoPipeline:
    """Verifica a utilização do Agent como orquestrador do pipeline."""

    def test_interface_deve_delegar_pipeline_ao_agent(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
        resultado_mock: MagicMock,
    ) -> None:
        """A execução da análise deve passar por Agent.executar_pipeline."""
        agent_mock.executar_pipeline.return_value = resultado_mock

        resultado = agent_mock.executar_pipeline(dados)

        agent_mock.executar_pipeline.assert_called_once_with(dados)
        assert resultado is resultado_mock

    def test_pipeline_nao_deve_chamar_llm(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
        resultado_mock: MagicMock,
    ) -> None:
        """Executar o pipeline não deve depender de perguntar()."""
        agent_mock.executar_pipeline.return_value = resultado_mock

        agent_mock.executar_pipeline(dados)

        agent_mock.perguntar.assert_not_called()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestRenderChat:
    """Testes do comportamento da aba de chat."""

    def test_chat_fica_indisponivel_sem_api_key(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
    ) -> None:
        """Sem API key, o chat deve mostrar aviso e não chamar o Agent."""
        from data_analysis_agent.interfaces.streamlit_app.app import _render_chat

        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = None

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.warning"
        ) as mock_warning:
            _render_chat(agent_mock, dados, "dados.csv")

        mock_warning.assert_called_once()
        agent_mock.perguntar.assert_not_called()

    def test_chat_chama_agent_com_pergunta(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
    ) -> None:
        """Uma pergunta enviada pelo usuário deve ser delegada ao Agent."""
        from data_analysis_agent.interfaces.streamlit_app.app import _render_chat

        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = "chave-teste"

        session_state = {
            "historico_chat": [],
        }

        agent_mock.perguntar.return_value = "Há uma forte correlação entre as colunas."

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.session_state",
            session_state,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.caption"
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.chat_input",
            return_value="Existe alguma correlação?",
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.chat_message"
        ) as mock_chat_message, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.spinner"
        ):
            _render_chat(agent_mock, dados, "dados.csv")

        agent_mock.perguntar.assert_called_once_with(
            "Existe alguma correlação?",
            dados,
            "dados.csv",
        )

        assert len(session_state["historico_chat"]) == 2
        assert session_state["historico_chat"][0] == {
            "role": "user",
            "content": "Existe alguma correlação?",
        }
        assert session_state["historico_chat"][1] == {
            "role": "assistant",
            "content": "Há uma forte correlação entre as colunas.",
        }

        assert mock_chat_message.called


# ---------------------------------------------------------------------------
# Separação entre pipeline e chat
# ---------------------------------------------------------------------------


class TestSeparacaoPipelineChat:
    """Garante que pipeline e chat continuam independentes."""

    def test_executar_pipeline_nao_chama_perguntar(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
        resultado_mock: MagicMock,
    ) -> None:
        """Executar a análise não deve iniciar uma chamada ao LLM."""
        agent_mock.executar_pipeline.return_value = resultado_mock

        agent_mock.executar_pipeline(dados)

        agent_mock.executar_pipeline.assert_called_once_with(dados)
        agent_mock.perguntar.assert_not_called()

    def test_perguntar_nao_executa_pipeline_automaticamente(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
    ) -> None:
        """O chat deve chamar perguntar(), sem executar o pipeline por conta própria."""
        agent_mock.perguntar.return_value = "Resposta."

        agent_mock.perguntar(
            "Qual é a média?",
            dados,
            "dados.csv",
        )

        agent_mock.perguntar.assert_called_once_with(
            "Qual é a média?",
            dados,
            "dados.csv",
        )
        agent_mock.executar_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Renderização de gráficos
# ---------------------------------------------------------------------------


class TestRenderGraficos:
    """Testes da renderização dos ChartSpec."""

    def test_mostra_mensagem_quando_nao_ha_graficos(self) -> None:
        """Nenhum gráfico deve resultar em uma mensagem informativa."""
        from data_analysis_agent.interfaces.streamlit_app.app import _render_graficos

        resultado = MagicMock()
        resultado.graficos = []

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.caption"
        ) as mock_caption:
            _render_graficos(resultado)

        mock_caption.assert_called_once_with(
            "Nenhum gráfico foi gerado para este dataset."
        )

    def test_renderiza_grafico_quando_existe_chart_spec(self) -> None:
        """Um ChartSpec deve ser convertido em Figure e renderizado."""
        from data_analysis_agent.interfaces.streamlit_app.app import _render_graficos

        chart = MagicMock()
        chart.titulo = "Distribuição de idade"
        chart.descricao = "Boxplot."
        chart.figura = {
            "data": [],
            "layout": {},
        }

        resultado = MagicMock()
        resultado.graficos = [chart]

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.subheader"
        ) as mock_subheader, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.caption"
        ) as mock_caption, patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.plotly_chart"
        ) as mock_plotly:
            _render_graficos(resultado)

        mock_subheader.assert_called_once_with("Distribuição de idade")
        mock_caption.assert_called_once_with("Boxplot.")
        mock_plotly.assert_called_once()


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------


class TestEstadoSessao:
    """Testes relacionados ao histórico do chat."""

    def test_historico_chat_e_inicializado_quando_ausente(
        self,
        agent_mock: MagicMock,
        dados: pd.DataFrame,
    ) -> None:
        """A aba de chat deve criar o histórico quando ele ainda não existe."""
        from data_analysis_agent.interfaces.streamlit_app.app import _render_chat

        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = "chave-teste"

        session_state: dict[str, object] = {}

        with patch(
            "data_analysis_agent.interfaces.streamlit_app.app.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.session_state",
            session_state,
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.caption"
        ), patch(
            "data_analysis_agent.interfaces.streamlit_app.app.st.chat_input",
            return_value=None,
        ):
            _render_chat(agent_mock, dados, "dados.csv")

        assert "historico_chat" in session_state
        assert session_state["historico_chat"] == []