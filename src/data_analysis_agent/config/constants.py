"""Constantes e valores padrão (limiares e defaults) usados pelo motor de análise.

Nenhuma lógica é executada aqui — apenas valores constantes referenciados
pelos módulos do `engine` quando sua implementação for feita.
"""
from __future__ import annotations

from typing import Final

IQR_MULTIPLICADOR_PADRAO: Final[float] = 1.5
"""Multiplicador padrão aplicado ao intervalo interquartil (IQR) na detecção de outliers."""

ZSCORE_LIMIAR_PADRAO: Final[float] = 3.0
"""Limiar padrão do escore-Z (em desvios-padrão) na detecção de outliers."""

LIMIAR_CORRELACAO_FORTE: Final[float] = 0.7
"""Valor absoluto mínimo de correlação para ser considerada 'forte'."""

ENCODING_PADRAO: Final[str] = "utf-8"
"""Encoding padrão utilizado na leitura de arquivos de dados."""
