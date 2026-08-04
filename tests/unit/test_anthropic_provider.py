"""Testes unitários para AnthropicProvider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from data_analysis_agent.config.settings import Settings
from data_analysis_agent.exceptions.errors import (
    ConfigurationError,
    LLMProviderError,
)
from data_analysis_agent.llm.anthropic_provider import AnthropicProvider


@pytest.fixture
def settings():
    """Configuração válida para os testes."""
    return Settings(
        anthropic_api_key="fake-api-key",
        anthropic_model="claude-sonnet-5",
    )


class TestInicializacao:
    """Testes do construtor."""

    def test_cria_cliente_anthropic(self, settings):
        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:

            provider = AnthropicProvider(settings)

            mock_client.assert_called_once_with(api_key="fake-api-key")
            assert provider is not None

    def test_erro_sem_api_key(self, settings):
        settings.anthropic_api_key = ""

        with pytest.raises(ConfigurationError):
            AnthropicProvider(settings)


class TestGeracaoInsights:
    """Testes do método gerar_insights."""

    def test_retorna_texto(self, settings):
        texto = "Insight gerado."

        resposta = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=texto)
            ]
        )

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:

            mock_client.return_value.messages.create.return_value = resposta

            provider = AnthropicProvider(settings)

            resultado = provider.gerar_insights(
                {"estatisticas_descritivas": {}}
            )

            assert resultado == texto

    def test_concatena_blocos_texto(self, settings):
        resposta = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Primeira linha."),
                SimpleNamespace(type="text", text="Segunda linha."),
            ]
        )

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:

            mock_client.return_value.messages.create.return_value = resposta

            provider = AnthropicProvider(settings)

            resultado = provider.gerar_insights({})

            assert resultado == "Primeira linha.\nSegunda linha."

    def test_converte_api_error(self, settings):
        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:

            erro = anthropic.APIError(
                message="erro",
                request=MagicMock(),
                body={},
            )

            mock_client.return_value.messages.create.side_effect = erro

            provider = AnthropicProvider(settings)

            with pytest.raises(LLMProviderError):
                provider.gerar_insights({})