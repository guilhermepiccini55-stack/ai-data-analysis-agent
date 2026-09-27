"""Agente conversacional e orquestrador central da aplicação.

Fase 3 do SDD: chat com tool-calling. Fase 4: o ``Agent`` passa a ser o
ponto de entrada único — tanto para rodar o pipeline direto
(:meth:`executar_pipeline`, sem exigir LLM) quanto para o chat com
tool-calling (:meth:`perguntar`, que exige um provider de LLM). O client
Anthropic é instanciado de forma lazy, só na primeira chamada a
:meth:`perguntar` — assim o ``Agent`` pode ser usado como orquestrador
mesmo sem ``DAA_ANTHROPIC_API_KEY`` configurada, desde que o chat não
seja usado.

Cada chamada a :meth:`perguntar` é independente — sem histórico de
conversa mantido entre chamadas (decisão de escopo da Fase 3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from data_analysis_agent.config.settings import Settings, obter_settings
from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.exceptions.errors import ConfigurationError, DataAnalysisAgentError
from data_analysis_agent.models.analysis_models import AnalysisResult

_MAX_ITERACOES = 6

_PROMPT_SISTEMA = (
    "Você é um assistente de análise de dados. Você tem acesso a "
    "ferramentas que executam etapas de um pipeline de análise (limpeza, "
    "análise estatística, visualização, relatório e insights). Use as "
    "ferramentas necessárias, na ordem correta, para responder à "
    "pergunta do usuário sobre o dataset fornecido. Não invente "
    "resultados — responda apenas com base no que as ferramentas "
    "retornarem."
)

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "limpar_dados",
        "description": (
            "Executa a limpeza dos dados brutos: remove duplicatas, trata "
            "valores ausentes, infere tipos. Deve ser chamada antes de "
            "analisar_dados ou gerar_visualizacoes."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analisar_dados",
        "description": (
            "Executa a análise estatística sobre os dados já limpos "
            "(estatísticas descritivas, outliers, correlações). Requer "
            "que limpar_dados já tenha sido chamada antes, nesta mesma "
            "conversa."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "gerar_visualizacoes",
        "description": (
            "Gera gráficos (boxplots, dispersão) a partir dos dados "
            "limpos e do resumo da análise. Requer limpar_dados e "
            "analisar_dados já executadas antes, nesta mesma conversa."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "gerar_relatorio",
        "description": (
            "Gera o relatório final em Markdown a partir dos resultados "
            "já produzidos. Requer limpar_dados, analisar_dados e "
            "gerar_visualizacoes já executadas antes, nesta mesma "
            "conversa."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "gerar_insights",
        "description": (
            "Gera insights textuais via LLM a partir do resumo da "
            "análise estatística. Requer analisar_dados já executada "
            "antes. Só funciona se um provider de LLM tiver sido "
            "configurado na engine — se falhar por falta de "
            "configuração, informe isso ao usuário em vez de tentar de "
            "novo."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class Agent:
    """Orquestrador central: expõe a AnalysisEngine tanto direto quanto via chat.

    Dois modos de uso, sob a mesma instância:

    - :meth:`executar_pipeline` — delega direto para
      ``AnalysisEngine.executar_pipeline()``. Não exige LLM configurado.
    - :meth:`perguntar` — chat com tool-calling sobre os métodos
      granulares da engine. Exige ``DAA_ANTHROPIC_API_KEY`` configurada;
      a checagem e a criação do client Anthropic são lazy (só acontecem
      na primeira chamada a este método).

    Cada chamada a :meth:`perguntar` é independente: não há histórico de
    conversa mantido entre chamadas. O estado intermediário do pipeline
    (DataFrame limpo, resumo de análise, gráficos etc.) vive apenas
    durante uma única chamada — nunca é persistido nem enviado ao LLM (só
    descrições textuais/JSON dos resultados são devolvidas como
    ``tool_result``).
    """

    def __init__(self, engine: AnalysisEngine, settings: Settings | None = None) -> None:
        """Inicializa o agente. Não exige API key configurada.

        Args:
            engine: instância de ``AnalysisEngine`` já configurada (com
                ``llm_provider`` injetado, se a tool ``gerar_insights``
                for necessária).
            settings: configurações da aplicação. Se omitido, são
                carregadas via :func:`obter_settings`.
        """
        self._engine = engine
        self._settings = settings or obter_settings()
        self._client: anthropic.Anthropic | None = None

    def executar_pipeline(self, dados: pd.DataFrame | str | Path) -> AnalysisResult:
        """Executa o pipeline completo, delegando direto para a AnalysisEngine.

        Modo não conversacional do orquestrador — não exige LLM
        configurado, já que não envolve tool-calling.

        Args:
            dados: ``DataFrame`` já carregado ou caminho para um arquivo
                (ex.: CSV) a ser carregado.

        Returns:
            AnalysisResult: resultado completo e autossuficiente da análise.
        """
        return self._engine.executar_pipeline(dados)

    def _obter_client(self) -> anthropic.Anthropic:
        """Instancia o client Anthropic sob demanda (lazy), na primeira chamada.

        Returns:
            anthropic.Anthropic: client já configurado, reaproveitado em
            chamadas seguintes a :meth:`perguntar`.

        Raises:
            ConfigurationError: se nenhuma chave de API estiver configurada.
        """
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise ConfigurationError(
                    "DAA_ANTHROPIC_API_KEY não configurada — necessária para usar perguntar()."
                )
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def perguntar(
        self,
        pergunta: str,
        dados: pd.DataFrame,
        fonte_dados: str = "DataFrame fornecido diretamente",
    ) -> str:
        """Responde a uma pergunta sobre o dataset, usando tool-calling.

        Args:
            pergunta: pergunta em linguagem natural do usuário.
            dados: ``DataFrame`` bruto sobre o qual as tools vão operar.
            fonte_dados: identificação da fonte de dados, usada apenas se
                a tool ``gerar_relatorio`` for chamada.

        Returns:
            str: resposta final do agente, em texto.

        Raises:
            ConfigurationError: se nenhuma chave de API estiver configurada
                (checagem lazy, feita aqui e não no construtor).
        """
        client = self._obter_client()
        estado: dict[str, Any] = {"dados_brutos": dados, "fonte_dados": fonte_dados}
        mensagens: list[dict[str, Any]] = [{"role": "user", "content": pergunta}]

        for _ in range(_MAX_ITERACOES):
            resposta = client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=1024,
                system=_PROMPT_SISTEMA,
                tools=_TOOLS,
                messages=mensagens,
            )

            if resposta.stop_reason != "tool_use":
                blocos_texto = [bloco.text for bloco in resposta.content if bloco.type == "text"]
                return "\n".join(blocos_texto)

            mensagens.append({"role": "assistant", "content": resposta.content})

            resultados_tools: list[dict[str, Any]] = []
            for bloco in resposta.content:
                if bloco.type != "tool_use":
                    continue
                resultado_tool = self._executar_tool(bloco.name, estado)
                resultados_tools.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": bloco.id,
                        "content": resultado_tool,
                    }
                )

            mensagens.append({"role": "user", "content": resultados_tools})

        return (
            "Não consegui concluir a análise dentro do número máximo de "
            "etapas permitidas. Tente reformular a pergunta."
        )

    def _executar_tool(self, nome_tool: str, estado: dict[str, Any]) -> str:
        """Executa uma tool pelo nome, delegando para a AnalysisEngine.

        Args:
            nome_tool: nome da tool solicitada pelo modelo.
            estado: dict mutável com os resultados intermediários já
                produzidos nesta chamada de :meth:`perguntar`.

        Returns:
            str: descrição textual (ou erro) do resultado da tool, para
            ser devolvida ao modelo como ``tool_result``.
        """
        try:
            if nome_tool == "limpar_dados":
                dados_limpos, relatorio_limpeza = self._engine.limpar(estado["dados_brutos"])
                estado["dados_limpos"] = dados_limpos
                estado["relatorio_limpeza"] = relatorio_limpeza
                return (
                    f"Limpeza concluída. Formato: {relatorio_limpeza.formato_original} -> "
                    f"{relatorio_limpeza.formato_final}. Duplicatas removidas: "
                    f"{relatorio_limpeza.duplicatas_removidas}."
                )

            if nome_tool == "analisar_dados":
                if "dados_limpos" not in estado:
                    return "Erro: chame limpar_dados antes de analisar_dados."
                resumo_analise = self._engine.analisar(estado["dados_limpos"])
                estado["resumo_analise"] = resumo_analise
                return resumo_analise.model_dump_json()

            if nome_tool == "gerar_visualizacoes":
                if "dados_limpos" not in estado or "resumo_analise" not in estado:
                    return (
                        "Erro: chame limpar_dados e analisar_dados antes de "
                        "gerar_visualizacoes."
                    )
                graficos = self._engine.visualizar(estado["dados_limpos"], estado["resumo_analise"])
                estado["graficos"] = graficos
                titulos = ", ".join(grafico.titulo for grafico in graficos) or "nenhum"
                return f"{len(graficos)} gráfico(s) gerado(s): {titulos}."

            if nome_tool == "gerar_relatorio":
                faltando = [
                    chave
                    for chave in ("relatorio_limpeza", "resumo_analise", "graficos")
                    if chave not in estado
                ]
                if faltando:
                    return (
                        "Erro: chame limpar_dados, analisar_dados e "
                        "gerar_visualizacoes antes de gerar_relatorio."
                    )
                resultado_parcial = AnalysisResult(
                    fonte_dados=estado["fonte_dados"],
                    relatorio_limpeza=estado["relatorio_limpeza"],
                    resultado_analise=estado["resumo_analise"],
                    graficos=estado["graficos"],
                    relatorio=None,
                )
                relatorio = self._engine.gerar_relatorio(resultado_parcial)
                estado["relatorio"] = relatorio
                return relatorio.markdown

            if nome_tool == "gerar_insights":
                if "resumo_analise" not in estado:
                    return "Erro: chame analisar_dados antes de gerar_insights."
                return self._engine.gerar_insights(estado["resumo_analise"])

            return f"Erro: tool desconhecida '{nome_tool}'."

        except DataAnalysisAgentError as exc:
            return f"Erro ao executar {nome_tool}: {exc.mensagem}"