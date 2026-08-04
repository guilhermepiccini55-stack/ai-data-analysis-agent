"""Provider concreto de LLM usando a API da Anthropic (Claude).

Único provider concreto desta fase (Seção 10 do SDD v2.1) — um segundo
provider só nasce quando houver necessidade real de trocar.
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from data_analysis_agent.config.settings import Settings, obter_settings
from data_analysis_agent.exceptions.errors import ConfigurationError, LLMProviderError

_PROMPT_SISTEMA = (
    "Você é um analista de dados. Você vai receber um resumo estruturado "
    "(JSON) com estatísticas descritivas, outliers e correlações de um "
    "dataset. Gere um parágrafo curto e objetivo, em português, "
    "destacando os achados mais relevantes. Não invente números que não "
    "estejam presentes no resumo fornecido."
)


class AnthropicProvider:
    """Provider de LLM que usa a API da Anthropic para gerar insights.

    Implementa implicitamente o Protocol ``LLMProvider`` (duck typing —
    sem necessidade de herança explícita).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Inicializa o provider a partir das configurações da aplicação.

        Args:
            settings: configurações já carregadas. Se omitido, são
                carregadas via :func:`obter_settings`.

        Raises:
            ConfigurationError: se nenhuma chave de API estiver configurada.
        """
        self._settings = settings or obter_settings()
        if not self._settings.anthropic_api_key:
            raise ConfigurationError(
                "DAA_ANTHROPIC_API_KEY não configurada — necessária para "
                "usar o AnthropicProvider."
            )
        self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)

    def gerar_insights(self, resumo: dict[str, Any]) -> str:
        """Gera um texto de insights a partir de um resumo estruturado.

        Args:
            resumo: dict serializável derivado de ``AnalysisSummary``
                (ex.: ``AnalysisSummary.model_dump()``).

        Returns:
            str: texto gerado pelo modelo com insights sobre os dados.

        Raises:
            LLMProviderError: se a chamada à API da Anthropic falhar.
        """
        try:
            mensagem = self._client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=1024,
                system=_PROMPT_SISTEMA,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(resumo, ensure_ascii=False, default=str),
                    }
                ],
            )
        except anthropic.APIError as exc:
            raise LLMProviderError(f"Falha ao chamar a API da Anthropic: {exc}") from exc

        blocos_texto = [bloco.text for bloco in mensagem.content if bloco.type == "text"]
        return "\n".join(blocos_texto)