"""Testes unitários complementares do módulo ``llm/``.

Este arquivo não repete o que já é coberto por ``test_llm_base.py`` (existência
do Protocol) nem por ``test_anthropic_provider.py`` (inicialização básica,
concatenação de blocos de texto e conversão de ``anthropic.APIError`` em
``LLMProviderError``). Aqui o foco é:

1. O contrato estrutural do Protocol ``LLMProvider`` (Seção 5 do SDD: providers
   são conectados por duck typing, sem herança explícita) — incluindo a
   verificação de que ``AnthropicProvider`` o satisfaz estruturalmente.
2. Detalhes do contrato de chamada da API da Anthropic feito por
   ``AnthropicProvider.gerar_insights`` que não são exercitados pelos testes
   existentes: parâmetros exatos enviados (`model`, `max_tokens`, `system`),
   a serialização do resumo via ``json.dumps(..., ensure_ascii=False,
   default=str)`` e o comportamento do construtor quando nenhuma
   ``Settings`` é passada explicitamente.
3. Casos de borda no processamento da resposta da API (conteúdo vazio,
   blocos não textuais misturados com blocos de texto).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from data_analysis_agent.config.settings import Settings
from data_analysis_agent.exceptions.errors import LLMProviderError
from data_analysis_agent.llm import anthropic_provider as anthropic_provider_module
from data_analysis_agent.llm.anthropic_provider import (
    _PROMPT_SISTEMA,
    AnthropicProvider,
)
from data_analysis_agent.llm.base import LLMProvider


@pytest.fixture
def settings() -> Settings:
    """Configuração válida para os testes (mesmo padrão de test_anthropic_provider.py)."""
    return Settings(
        anthropic_api_key="fake-api-key",
        anthropic_model="claude-sonnet-5",
    )


def _resposta_com_blocos(*blocos: SimpleNamespace) -> SimpleNamespace:
    """Monta um objeto de resposta falso no mesmo formato usado pelo SDK."""
    return SimpleNamespace(content=list(blocos))


# ---------------------------------------------------------------------------
# Contrato estrutural do Protocol LLMProvider (base.py)
# ---------------------------------------------------------------------------


class TestContratoLLMProvider:
    """O Protocol define apenas a assinatura de ``gerar_insights``.

    Conforme a Seção 5 do SDD, qualquer provider — inclusive o
    ``AnthropicProvider`` — se conecta a ele por duck typing (compatibilidade
    estrutural), nunca por herança explícita.
    """

    def test_protocolo_nao_pode_ser_instanciado_diretamente(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()

    def test_classe_com_gerar_insights_satisfaz_o_protocolo(self) -> None:
        class ProviderFake:
            def gerar_insights(self, resumo: dict) -> str:
                return "ok"

        assert isinstance(ProviderFake(), LLMProvider)

    def test_classe_sem_gerar_insights_nao_satisfaz_o_protocolo(self) -> None:
        class ProviderIncompleto:
            def outro_metodo(self) -> None:
                ...

        assert not isinstance(ProviderIncompleto(), LLMProvider)

    def test_anthropic_provider_satisfaz_o_protocolo_por_duck_typing(
        self, settings: Settings
    ) -> None:
        """``AnthropicProvider`` não herda de ``LLMProvider`` no código-fonte,
        mas deve ser reconhecido como tal estruturalmente — é isso que permite
        à ``AnalysisEngine`` (Seção 5 do SDD) aceitar qualquer provider
        injetado sem acoplamento por herança.
        """
        assert LLMProvider not in AnthropicProvider.__mro__

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ):
            provider = AnthropicProvider(settings)

        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Construtor: uso de obter_settings() quando nenhuma Settings é passada
# ---------------------------------------------------------------------------


class TestInicializacaoComSettingsPadrao:
    def test_usa_obter_settings_quando_settings_nao_e_fornecida(
        self, settings: Settings
    ) -> None:
        with (
            patch.object(
                anthropic_provider_module, "obter_settings", return_value=settings
            ) as mock_obter_settings,
            patch(
                "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
            ) as mock_client,
        ):
            provider = AnthropicProvider()

            mock_obter_settings.assert_called_once_with()
            mock_client.assert_called_once_with(api_key="fake-api-key")
            assert provider is not None

    def test_settings_explicita_tem_precedencia_sobre_obter_settings(
        self, settings: Settings
    ) -> None:
        with (
            patch.object(anthropic_provider_module, "obter_settings") as mock_obter_settings,
            patch("data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"),
        ):
            AnthropicProvider(settings)

            mock_obter_settings.assert_not_called()


# ---------------------------------------------------------------------------
# Contrato exato da chamada a messages.create
# ---------------------------------------------------------------------------


class TestContratoDaChamadaApi:
    def test_envia_model_max_tokens_e_prompt_de_sistema_corretos(
        self, settings: Settings
    ) -> None:
        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_create = mock_client.return_value.messages.create
            mock_create.return_value = _resposta_com_blocos(
                SimpleNamespace(type="text", text="ok")
            )

            provider = AnthropicProvider(settings)
            provider.gerar_insights({"estatisticas_descritivas": {}})

            _, kwargs = mock_create.call_args
            assert kwargs["model"] == settings.anthropic_model
            assert kwargs["max_tokens"] == 1024
            assert kwargs["system"] == _PROMPT_SISTEMA

    def test_envia_resumo_serializado_como_json_no_content_da_mensagem(
        self, settings: Settings
    ) -> None:
        resumo = {"estatisticas_descritivas": {"idade": {"media": 30}}, "outliers": []}

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_create = mock_client.return_value.messages.create
            mock_create.return_value = _resposta_com_blocos(
                SimpleNamespace(type="text", text="ok")
            )

            provider = AnthropicProvider(settings)
            provider.gerar_insights(resumo)

            _, kwargs = mock_create.call_args
            mensagens = kwargs["messages"]
            assert len(mensagens) == 1
            assert mensagens[0]["role"] == "user"
            assert json.loads(mensagens[0]["content"]) == resumo

    def test_serializa_caracteres_unicode_sem_escapar(self, settings: Settings) -> None:
        """``ensure_ascii=False``: acentos devem chegar intactos no payload,
        e não como sequências de escape (``\\uXXXX``)."""
        resumo = {"observacao": "correlação forte entre idade e salário"}

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_create = mock_client.return_value.messages.create
            mock_create.return_value = _resposta_com_blocos(
                SimpleNamespace(type="text", text="ok")
            )

            provider = AnthropicProvider(settings)
            provider.gerar_insights(resumo)

            conteudo_enviado = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "correlação" in conteudo_enviado
            assert "\\u" not in conteudo_enviado

    def test_serializa_valores_nao_json_nativos_via_default_str(
        self, settings: Settings
    ) -> None:
        """``default=str``: tipos que o ``json`` não sabe serializar nativamente
        (ex.: ``datetime``, vindo de campos como ``AnalysisResult.gerado_em``)
        devem ser convertidos via ``str()`` em vez de levantar ``TypeError``."""
        from datetime import datetime

        momento = datetime(2024, 1, 1, 12, 30, 0)
        resumo = {"gerado_em": momento}

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_create = mock_client.return_value.messages.create
            mock_create.return_value = _resposta_com_blocos(
                SimpleNamespace(type="text", text="ok")
            )

            provider = AnthropicProvider(settings)
            # Não deve levantar TypeError por causa do campo datetime.
            provider.gerar_insights(resumo)

            conteudo_enviado = mock_create.call_args.kwargs["messages"][0]["content"]
            assert json.loads(conteudo_enviado) == {"gerado_em": str(momento)}


# ---------------------------------------------------------------------------
# Processamento da resposta da API
# ---------------------------------------------------------------------------


class TestProcessamentoDaResposta:
    def test_resposta_sem_blocos_retorna_string_vazia(self, settings: Settings) -> None:
        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_client.return_value.messages.create.return_value = _resposta_com_blocos()

            provider = AnthropicProvider(settings)
            resultado = provider.gerar_insights({})

            assert resultado == ""

    def test_ignora_blocos_que_nao_sao_do_tipo_text(self, settings: Settings) -> None:
        """A resposta pode conter blocos de outros tipos (ex.: ``tool_use``);
        apenas blocos ``text`` devem compor o resultado final."""
        resposta = _resposta_com_blocos(
            SimpleNamespace(type="tool_use", input={"foo": "bar"}),
            SimpleNamespace(type="text", text="Único texto relevante."),
        )

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_client.return_value.messages.create.return_value = resposta

            provider = AnthropicProvider(settings)
            resultado = provider.gerar_insights({})

            assert resultado == "Único texto relevante."


# ---------------------------------------------------------------------------
# Encadeamento de erros (complementar a test_converte_api_error)
# ---------------------------------------------------------------------------


class TestEncadeamentoDeErros:
    def test_llm_provider_error_preserva_excecao_original_como_causa(
        self, settings: Settings
    ) -> None:
        """Além de converter para ``LLMProviderError`` (já testado em
        test_anthropic_provider.py), a exceção original da Anthropic deve
        ficar preservada em ``__cause__`` para não perder o traceback/causa
        raiz na hora de depurar falhas de integração."""
        erro_original = anthropic.APIError(
            message="serviço indisponível",
            request=MagicMock(),
            body={},
        )

        with patch(
            "data_analysis_agent.llm.anthropic_provider.anthropic.Anthropic"
        ) as mock_client:
            mock_client.return_value.messages.create.side_effect = erro_original

            provider = AnthropicProvider(settings)

            with pytest.raises(LLMProviderError) as exc_info:
                provider.gerar_insights({})

            assert exc_info.value.__cause__ is erro_original