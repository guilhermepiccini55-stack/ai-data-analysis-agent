"""Interface Streamlit do AI Data Analysis Agent.

Camada de interface — fina, sem lógica de negócio própria. Cada
interação do usuário aciona a ``AnalysisEngine`` (stateless) e o
resultado é guardado em ``st.session_state``: essa é uma responsabilidade
explícita da interface, não da engine (Seção 5 do SDD v2.0).

Estrutura desta primeira versão: single-page com abas (``st.tabs``).
Nada foi extraído para ``components/`` ainda — só quando houver
repetição real que justifique a abstração (YAGNI).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.config.settings import obter_settings
from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.exceptions.errors import DataAnalysisAgentError
from data_analysis_agent.exporters.html import HTMLExporter
from data_analysis_agent.exporters.pdf import PDFExporter
from data_analysis_agent.llm.anthropic_provider import AnthropicProvider
from data_analysis_agent.models.analysis_models import AnalysisResult

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")


def _render_limpeza(resultado: AnalysisResult) -> None:
    """Renderiza a aba de limpeza de dados: métricas, perfil de colunas, imputações."""
    relatorio = resultado.relatorio_limpeza

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Linhas (antes → depois)",
        f"{relatorio.formato_original[0]} → {relatorio.formato_final[0]}",
    )
    col2.metric(
        "Colunas (antes → depois)",
        f"{relatorio.formato_original[1]} → {relatorio.formato_final[1]}",
    )
    col3.metric("Duplicatas removidas", relatorio.duplicatas_removidas)

    st.subheader("Perfil das colunas")
    if relatorio.colunas_perfil:
        df_perfil = pd.DataFrame(
            [perfil.model_dump() for perfil in relatorio.colunas_perfil]
        )
        st.dataframe(df_perfil, width="stretch")
    else:
        st.caption("Nenhum perfil de coluna disponível.")

    if relatorio.valores_imputados:
        st.subheader("Valores imputados por coluna")
        st.dataframe(
            pd.DataFrame(
                relatorio.valores_imputados.items(),
                columns=["Coluna", "Valores imputados"],
            ),
            width="stretch",
        )

    if relatorio.conversoes_tipo:
        st.subheader("Conversões de tipo")
        st.dataframe(
            pd.DataFrame(
                relatorio.conversoes_tipo.items(),
                columns=["Coluna", "Conversão"],
            ),
            width="stretch",
        )


def _render_analise(resultado: AnalysisResult) -> None:
    """Renderiza a aba de análise estatística: estatísticas, outliers, correlações."""
    resumo = resultado.resultado_analise

    st.subheader("Estatísticas descritivas")
    if resumo.estatisticas_descritivas:
        st.dataframe(
            pd.DataFrame(resumo.estatisticas_descritivas),
            width="stretch",
        )
    else:
        st.caption("Nenhuma estatística descritiva disponível.")

    st.subheader("Outliers")
    if resumo.outliers:
        df_outliers = pd.DataFrame(
            [
                {
                    "Coluna": outlier.coluna,
                    "Método": outlier.metodo.value,
                    "Limite inferior": outlier.limite_inferior,
                    "Limite superior": outlier.limite_superior,
                    "Total de outliers": outlier.total_outliers,
                }
                for outlier in resumo.outliers
            ]
        )
        st.dataframe(df_outliers, width="stretch")
    else:
        st.caption("Nenhum outlier detectado.")

    st.subheader("Correlações")
    if resumo.correlacoes is not None:
        st.write(f"Método: `{resumo.correlacoes.metodo}`")
        st.dataframe(
            pd.DataFrame(resumo.correlacoes.matriz_correlacao),
            width="stretch",
        )
        if resumo.correlacoes.pares_fortemente_correlacionados:
            st.subheader("Pares fortemente correlacionados")
            st.dataframe(
                pd.DataFrame(
                    resumo.correlacoes.pares_fortemente_correlacionados,
                    columns=["Coluna A", "Coluna B", "Correlação"],
                ),
                width="stretch",
            )
    else:
        st.caption("Correlação não aplicável (dados insuficientes).")


def _render_graficos(resultado: AnalysisResult) -> None:
    """Renderiza a aba de visualizações, reconstruindo cada ``ChartSpec`` como Figure.

    Conforme a Seção 7 do SDD v2.0, ``ChartSpec.figura`` é um ``dict``
    serializável — só quem precisa renderizar (esta interface) reconstrói
    o objeto ``plotly.graph_objects.Figure`` a partir dele.
    """
    graficos = resultado.graficos
    if not graficos:
        st.caption("Nenhum gráfico foi gerado para este dataset.")
        return

    for chart_spec in graficos:
        st.subheader(chart_spec.titulo)
        if chart_spec.descricao:
            st.caption(chart_spec.descricao)
        figura = go.Figure(chart_spec.figura)
        st.plotly_chart(figura, width="stretch")


def _render_relatorio(resultado: AnalysisResult, nome_arquivo: str | None) -> None:
    relatorio = resultado.relatorio
    if relatorio is None:
        st.caption("Nenhum relatório foi gerado para esta análise.")
        return

    nome_base = Path(nome_arquivo).stem if nome_arquivo else "analise"

    col_md, col_html, col_pdf = st.columns(3)
    col_md.download_button(
        label="⬇️ Baixar relatório (.md)",
        data=relatorio.markdown,
        file_name=f"relatorio_{nome_base}.md",
        mime="text/markdown",
    )
    col_html.download_button(
        label="⬇️ Baixar relatório (.html)",
        data=HTMLExporter().exportar(resultado),
        file_name=f"relatorio_{nome_base}.html",
        mime="text/html",
    )
    col_pdf.download_button(
        label="⬇️ Baixar relatório (.pdf)",
        data=PDFExporter().exportar(resultado),
        file_name=f"relatorio_{nome_base}.pdf",
        mime="application/pdf",
    )

    st.divider()
    st.markdown(relatorio.markdown)


@st.cache_resource
def _obter_agent() -> Agent:
    """Cria o ``Agent`` (orquestrador central) uma única vez por processo.

    Funciona mesmo sem ``DAA_ANTHROPIC_API_KEY`` configurada — a chave só
    é checada de forma lazy, na primeira chamada a ``agent.perguntar()``
    (Fase 4: unificação sob uma única API, sem forçar configuração de LLM
    para quem só quer rodar o pipeline).
    """
    settings = obter_settings()
    llm_provider = (
        AnthropicProvider(settings=settings)
        if settings.anthropic_api_key
        else None
    )
    engine = (
        AnalysisEngine(llm_provider=llm_provider)
        if llm_provider
        else AnalysisEngine()
    )
    return Agent(engine, settings=settings)


def _render_chat(
    agent: Agent,
    dados_brutos: pd.DataFrame,
    fonte_dados: str,
) -> None:
    if not obter_settings().anthropic_api_key:
        st.warning(
            "Chat indisponível: configure `DAA_ANTHROPIC_API_KEY` no `.env` "
            "para habilitar esta aba."
        )
        return

    st.caption(
        "Cada pergunta é respondida de forma independente — o agente não "
        "guarda memória de perguntas anteriores nesta primeira versão."
    )

    if "historico_chat" not in st.session_state:
        st.session_state["historico_chat"] = []

    for turno in st.session_state["historico_chat"]:
        with st.chat_message(turno["role"]):
            st.markdown(turno["content"])

    pergunta = st.chat_input("Pergunte algo sobre os dados...")
    if pergunta:
        st.session_state["historico_chat"].append(
            {"role": "user", "content": pergunta}
        )
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    resposta = agent.perguntar(
                        pergunta,
                        dados_brutos,
                        fonte_dados,
                    )
                except DataAnalysisAgentError as exc:
                    resposta = f"Erro: {exc.mensagem}"
            st.markdown(resposta)

        st.session_state["historico_chat"].append(
            {"role": "assistant", "content": resposta}
        )


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    if "resultado" not in st.session_state:
        st.session_state["resultado"] = None
    if "nome_arquivo" not in st.session_state:
        st.session_state["nome_arquivo"] = None
    if "dados_brutos" not in st.session_state:
        st.session_state["dados_brutos"] = None

    agent = _obter_agent()

    st.title("AI Data Analysis Agent")
    st.caption(
        "Análise automatizada de dados tabulares — limpeza, estatísticas e visualizações."
    )

    arquivo = st.file_uploader("Envie um arquivo CSV", type=["csv"])
    executar = st.button(
        "Executar análise",
        type="primary",
        disabled=arquivo is None,
    )

    if executar and arquivo is not None:
        with st.spinner("Executando pipeline de análise..."):
            try:
                dados = pd.read_csv(arquivo)
                st.session_state["resultado"] = agent.executar_pipeline(dados)
                st.session_state["nome_arquivo"] = arquivo.name
                st.session_state["dados_brutos"] = dados
                st.session_state["historico_chat"] = []
            except Exception as exc:  # noqa: BLE001
                st.session_state["resultado"] = None
                st.error(f"Falha ao executar o pipeline: {exc}")

    resultado: AnalysisResult | None = st.session_state["resultado"]

    if resultado is None:
        st.info(
            "Envie um arquivo CSV e clique em **Executar análise** para começar."
        )
        return

    if st.session_state["nome_arquivo"]:
        st.success(
            f"Análise concluída para **{st.session_state['nome_arquivo']}**"
        )

    aba_limpeza, aba_analise, aba_graficos, aba_relatorio, aba_chat = st.tabs(
        ["🧹 Limpeza", "📊 Análise", "📈 Gráficos", "📄 Relatório", "💬 Chat"]
    )

    with aba_limpeza:
        _render_limpeza(resultado)
    with aba_analise:
        _render_analise(resultado)
    with aba_graficos:
        _render_graficos(resultado)
    with aba_relatorio:
        _render_relatorio(resultado, st.session_state["nome_arquivo"])
    with aba_chat:
        _render_chat(
            agent,
            st.session_state["dados_brutos"],
            resultado.fonte_dados,
        )


main()