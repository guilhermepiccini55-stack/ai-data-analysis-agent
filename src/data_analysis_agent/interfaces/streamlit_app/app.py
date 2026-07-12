"""Ponto de entrada da interface Streamlit do AI Data Analysis Agent.

Fundação da Fase 1: nenhuma funcionalidade de limpeza, análise,
visualização, relatório ou integração com IA está implementada ainda.
Este arquivo apenas garante que a aplicação Streamlit pode ser executada,
exibindo uma página provisória. O dashboard interativo real (upload de
dados, exibição de resultados via `AnalysisEngine`, gerenciamento de
`AnalysisResult` em `st.session_state`) será implementado em uma etapa
posterior da Fase 1.
"""
from __future__ import annotations

import streamlit as st

from data_analysis_agent.config.logging_config import configurar_logging, obter_logger
from data_analysis_agent.config.settings import obter_settings

_logger = obter_logger(__name__)


def main() -> None:
    """Renderiza a página inicial provisória da aplicação Streamlit."""
    settings = obter_settings()
    configurar_logging(settings.nivel_log)
    _logger.info("Aplicação Streamlit inicializada (fundação da Fase 1).")

    st.set_page_config(page_title=settings.app_name, layout="wide")
    st.title(settings.app_name)
    st.info(
        "Fundação do projeto (Fase 1) em andamento. "
        "O dashboard interativo será implementado em uma etapa posterior."
    )


if __name__ == "__main__":
    main()
