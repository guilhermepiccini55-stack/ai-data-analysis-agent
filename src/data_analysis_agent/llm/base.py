"""Interface (Protocol) para providers de LLM.

Conforme a Seção 10 do SDD v2.1, a Fase 2 entrega apenas este Protocol
mais um único provider concreto. Um segundo provider só é criado quando
houver necessidade real de trocar (YAGNI).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Contrato que qualquer provider de LLM usado pela aplicação deve seguir.

    Recebe sempre dados já estruturados — nunca um ``DataFrame`` bruto ou
    um objeto de domínio completo — e devolve texto gerado pelo modelo.
    """

    def gerar_insights(self, resumo: dict[str, Any]) -> str:
        """Gera um texto de insights a partir de um resumo estruturado.

        Args:
            resumo: dict serializável derivado de ``AnalysisSummary``
                (ex.: ``AnalysisSummary.model_dump()``).

        Returns:
            str: texto gerado pelo LLM com insights sobre os dados.
        """
        ...