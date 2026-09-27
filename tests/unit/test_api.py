"""Testes unitários da API REST (FastAPI) — Fase 5.

Testes black-box dos endpoints, usando `TestClient` e substituindo a
dependência `obter_agent` por um `Agent` mockado via
`app.dependency_overrides` — sem instanciar `AnalysisEngine`/client
Anthropic reais e sem chamadas de rede.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.api.app import app
from data_analysis_agent.api.dependencies import obter_agent
from data_analysis_agent.exceptions.errors import ConfigurationError, DataCleaningError
from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _limpar_overrides_e_cache():
    """Garante isolamento entre testes: sem overrides/cache vazando entre eles."""
    obter_agent.cache_clear()
    yield
    app.dependency_overrides.clear()
    obter_agent.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def cleaning_report_fake() -> CleaningReport:
    return CleaningReport(
        formato_original=(10, 3),
        formato_final=(9, 3),
        duplicatas_removidas=1,
        colunas_perfil=[],
        valores_imputados={},
        conversoes_tipo={},
    )


@pytest.fixture
def analysis_result_fake(cleaning_report_fake: CleaningReport) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="DataFrame fornecido diretamente",
        relatorio_limpeza=cleaning_report_fake,
        resultado_analise=AnalysisSummary(estatisticas_descritivas={}, outliers=[], correlacoes=None),
        graficos=[],
        relatorio=None,
    )


@pytest.fixture
def csv_upload() -> tuple[str, io.BytesIO, str]:
    conteudo = b"idade,salario\n20,2000\n21,2500\n"
    return ("dados.csv", io.BytesIO(conteudo), "text/csv")


def _sobrescrever_agent(agent_mock: MagicMock) -> None:
    app.dependency_overrides[obter_agent] = lambda: agent_mock


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_retorna_ok(client: TestClient) -> None:
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /analises
# ---------------------------------------------------------------------------


def test_analises_executa_pipeline_com_sucesso(
    client: TestClient,
    csv_upload: tuple[str, io.BytesIO, str],
    analysis_result_fake: AnalysisResult,
) -> None:
    agent_mock = MagicMock(spec=Agent)
    agent_mock.executar_pipeline.return_value = analysis_result_fake
    _sobrescrever_agent(agent_mock)

    nome, arquivo, tipo = csv_upload
    resposta = client.post("/analises", files={"arquivo": (nome, arquivo, tipo)})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["fonte_dados"] == "DataFrame fornecido diretamente"
    assert corpo["relatorio_limpeza"]["duplicatas_removidas"] == 1

    agent_mock.executar_pipeline.assert_called_once()
    dados_enviados = agent_mock.executar_pipeline.call_args.args[0]
    assert isinstance(dados_enviados, pd.DataFrame)
    assert list(dados_enviados.columns) == ["idade", "salario"]


def test_analises_com_csv_invalido_retorna_400(client: TestClient) -> None:
    agent_mock = MagicMock(spec=Agent)
    _sobrescrever_agent(agent_mock)

    arquivo_vazio = io.BytesIO(b"")
    resposta = client.post("/analises", files={"arquivo": ("vazio.csv", arquivo_vazio, "text/csv")})

    assert resposta.status_code == 400
    agent_mock.executar_pipeline.assert_not_called()


def test_analises_propaga_erro_de_negocio_como_422(
    client: TestClient, csv_upload: tuple[str, io.BytesIO, str]
) -> None:
    agent_mock = MagicMock(spec=Agent)
    agent_mock.executar_pipeline.side_effect = DataCleaningError("dados malformados")
    _sobrescrever_agent(agent_mock)

    nome, arquivo, tipo = csv_upload
    resposta = client.post("/analises", files={"arquivo": (nome, arquivo, tipo)})

    assert resposta.status_code == 422
    corpo = resposta.json()
    assert corpo["erro"] == "DataCleaningError"
    assert corpo["mensagem"] == "dados malformados"


def test_analises_sem_arquivo_retorna_422_de_validacao(client: TestClient) -> None:
    """Sem o campo `arquivo`, o FastAPI deve rejeitar a requisição por validação."""
    agent_mock = MagicMock(spec=Agent)
    _sobrescrever_agent(agent_mock)

    resposta = client.post("/analises")

    assert resposta.status_code == 422
    agent_mock.executar_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# POST /perguntas
# ---------------------------------------------------------------------------


def test_perguntas_feliz(client: TestClient, csv_upload: tuple[str, io.BytesIO, str]) -> None:
    agent_mock = MagicMock(spec=Agent)
    agent_mock.perguntar.return_value = "A idade média é 20.5."
    _sobrescrever_agent(agent_mock)

    nome, arquivo, tipo = csv_upload
    resposta = client.post(
        "/perguntas",
        data={"pergunta": "Qual a idade média?"},
        files={"arquivo": (nome, arquivo, tipo)},
    )

    assert resposta.status_code == 200
    assert resposta.json() == {"resposta": "A idade média é 20.5."}

    agent_mock.perguntar.assert_called_once()
    args, kwargs = agent_mock.perguntar.call_args
    assert args[0] == "Qual a idade média?"
    assert isinstance(args[1], pd.DataFrame)
    assert kwargs["fonte_dados"] == "dados.csv"


def test_perguntas_sem_api_key_retorna_503(
    client: TestClient, csv_upload: tuple[str, io.BytesIO, str]
) -> None:
    agent_mock = MagicMock(spec=Agent)
    agent_mock.perguntar.side_effect = ConfigurationError(
        "DAA_ANTHROPIC_API_KEY não configurada — necessária para usar perguntar()."
    )
    _sobrescrever_agent(agent_mock)

    nome, arquivo, tipo = csv_upload
    resposta = client.post(
        "/perguntas",
        data={"pergunta": "Qual a idade média?"},
        files={"arquivo": (nome, arquivo, tipo)},
    )

    assert resposta.status_code == 503
    assert resposta.json()["erro"] == "ConfigurationError"


def test_perguntas_com_csv_invalido_retorna_400(client: TestClient) -> None:
    agent_mock = MagicMock(spec=Agent)
    _sobrescrever_agent(agent_mock)

    arquivo_vazio = io.BytesIO(b"")
    resposta = client.post(
        "/perguntas",
        data={"pergunta": "Qual a idade média?"},
        files={"arquivo": ("vazio.csv", arquivo_vazio, "text/csv")},
    )

    assert resposta.status_code == 400
    agent_mock.perguntar.assert_not_called()


# ---------------------------------------------------------------------------
# Injeção de dependências (api/dependencies.py) — sem chamadas de rede
# ---------------------------------------------------------------------------


class TestObterAgent:
    """Espelha os testes de `_obter_agent` da interface Streamlit (test_app.py)."""

    def test_obter_agent_cria_engine_sem_provider_quando_nao_ha_api_key(self) -> None:
        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = None

        engine_mock = MagicMock()
        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.api.dependencies.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.api.dependencies.AnalysisEngine",
            return_value=engine_mock,
        ) as mock_engine, patch(
            "data_analysis_agent.api.dependencies.Agent",
            return_value=agent_mock,
        ):
            resultado = obter_agent()

        mock_engine.assert_called_once_with()
        assert resultado is agent_mock

    def test_obter_agent_cria_provider_quando_ha_api_key(self) -> None:
        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = "chave-teste"

        provider_mock = MagicMock()
        engine_mock = MagicMock()
        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.api.dependencies.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.api.dependencies.AnthropicProvider",
            return_value=provider_mock,
        ) as mock_provider, patch(
            "data_analysis_agent.api.dependencies.AnalysisEngine",
            return_value=engine_mock,
        ) as mock_engine, patch(
            "data_analysis_agent.api.dependencies.Agent",
            return_value=agent_mock,
        ):
            resultado = obter_agent()

        mock_provider.assert_called_once_with(settings=settings_mock)
        mock_engine.assert_called_once_with(llm_provider=provider_mock)
        assert resultado is agent_mock

    def test_obter_agent_e_cacheado_entre_chamadas(self) -> None:
        settings_mock = MagicMock()
        settings_mock.anthropic_api_key = None
        agent_mock = MagicMock(spec=Agent)

        with patch(
            "data_analysis_agent.api.dependencies.obter_settings",
            return_value=settings_mock,
        ), patch(
            "data_analysis_agent.api.dependencies.AnalysisEngine",
        ), patch(
            "data_analysis_agent.api.dependencies.Agent",
            return_value=agent_mock,
        ) as mock_agent:
            primeira = obter_agent()
            segunda = obter_agent()

        assert primeira is segunda
        mock_agent.assert_called_once()
