"""Configuração centralizada do sistema de logging da aplicação.

Antecipado da Fase 4 para a Fase 1, conforme a Seção 8 do SDD v2.0.
"""
from __future__ import annotations

import logging
import sys

_FORMATO_PADRAO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configurar_logging(nivel: str = "INFO", formato: str | None = None) -> None:
    """Configura o logging raiz da aplicação.

    Deve ser chamada uma única vez, no ponto de entrada de cada interface
    (ex.: no início de ``app.py`` do Streamlit), e não pelos módulos do
    ``engine``, que apenas obtêm loggers via :func:`obter_logger`.

    Args:
        nivel: nível mínimo de log a ser exibido (ex.: "DEBUG", "INFO").
        formato: formato customizado das mensagens de log. Se ``None``,
            usa o formato padrão da aplicação.
    """
    logging.basicConfig(
        level=nivel.upper(),
        format=formato or _FORMATO_PADRAO,
        stream=sys.stdout,
        force=True,
    )


def obter_logger(nome: str) -> logging.Logger:
    """Retorna um logger nomeado, seguindo a configuração definida pela aplicação.

    Args:
        nome: nome do logger, tipicamente ``__name__`` do módulo chamador.

    Returns:
        logging.Logger: instância de logger pronta para uso.
    """
    return logging.getLogger(nome)
