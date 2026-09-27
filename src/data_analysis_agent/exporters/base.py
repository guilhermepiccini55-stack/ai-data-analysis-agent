"""Interface (Protocol) para exportadores de relatório.

Conforme a Seção 6 do prompt de Fase 6 (e no mesmo espírito de
``llm/base.py`` na Fase 2), esta camada nasce apenas agora — Fase 6 —
porque é só a partir de agora que existe mais de um formato real de
saída (Markdown, HTML, PDF) para o relatório final. Antes disso, a
abstração seria especulativa (YAGNI).

Cada implementação concreta recebe sempre o ``AnalysisResult`` completo
e devolve os bytes já prontos do formato correspondente — nunca reabre
ou reimplementa a lógica de geração do conteúdo, que continua vivendo
exclusivamente em ``engine/report.py`` (``gerar_relatorio_markdown``).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from data_analysis_agent.models.analysis_models import AnalysisResult


@runtime_checkable
class ReportExporter(Protocol):
    """Contrato que qualquer exportador de relatório deve seguir.

    Attributes:
        extensao: extensão de arquivo associada ao formato (sem o ponto,
            ex.: ``"md"``, ``"html"``, ``"pdf"``).
        mime_type: MIME type associado ao formato, usado por quem serve
            ou disponibiliza o arquivo para download (ex.: Streamlit).
    """

    extensao: str
    mime_type: str

    def exportar(self, resultado: AnalysisResult) -> bytes:
        """Exporta o relatório final a partir de um ``AnalysisResult``.

        Args:
            resultado: resultado completo do pipeline de análise, já
                contendo limpeza, análise estatística e visualizações.

        Returns:
            bytes: conteúdo do relatório já codificado no formato do
                exportador.
        """
        ...