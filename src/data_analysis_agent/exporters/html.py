"""Exportador de relatório em HTML.

Conforme a Seção 8 do prompt de Fase 6: reutiliza o Markdown já
produzido (via ``MarkdownExporter``, que por sua vez reutiliza
``gerar_relatorio_markdown`` — nenhuma lógica de conteúdo é duplicada
aqui) e o converte para HTML com a biblioteca ``markdown``. Os gráficos
Plotly são reconstruídos a partir do ``dict`` serializado em cada
``ChartSpec.figura`` (mesmo padrão já usado pela interface Streamlit em
``interfaces/streamlit_app/app.py``) e embutidos como HTML interativo.

``plotly`` só é importado aqui — nenhuma outra parte da camada de
exportação precisa dele, conforme a Seção 8 pede explicitamente.
"""
from __future__ import annotations

import html as html_stdlib

import markdown as markdown_lib
import plotly.graph_objects as go

from data_analysis_agent.exporters.markdown import MarkdownExporter
from data_analysis_agent.models.analysis_models import AnalysisResult
from data_analysis_agent.models.report_models import ChartSpec

_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>{titulo}</title>
</head>
<body>
{corpo}
{graficos}
</body>
</html>
"""


class HTMLExporter:
    """Exporta o relatório final como uma página HTML autossuficiente.

    O conteúdo textual vem do Markdown já existente, convertido para
    HTML. Quando há gráficos (``AnalysisResult.graficos``), eles são
    embutidos como Plotly interativo logo após o conteúdo textual.
    """

    extensao: str = "html"
    mime_type: str = "text/html"

    def __init__(self) -> None:
        self._markdown_exporter = MarkdownExporter()

    def exportar(self, resultado: AnalysisResult) -> bytes:
        """Gera o relatório em HTML e o codifica em bytes.

        Args:
            resultado: resultado completo do pipeline de análise.

        Returns:
            bytes: página HTML completa, codificada em UTF-8.
        """
        markdown_texto = self._markdown_exporter.exportar(resultado).decode("utf-8")
        corpo_html = markdown_lib.markdown(markdown_texto)

        titulo = resultado.relatorio.metadata.titulo if resultado.relatorio else resultado.fonte_dados
        graficos_html = self._renderizar_graficos(resultado.graficos)

        pagina = _TEMPLATE.format(
            titulo=html_stdlib.escape(titulo),
            corpo=corpo_html,
            graficos=graficos_html,
        )
        return pagina.encode("utf-8")

    def _renderizar_graficos(self, graficos: list[ChartSpec]) -> str:
        """Reconstrói cada ``ChartSpec`` como Plotly e devolve o HTML combinado.

        A biblioteca Plotly (``plotly.js``) é embutida inline (``include_plotlyjs
        ="inline"``) uma única vez, no primeiro gráfico, para que o HTML
        exportado seja autocontido e funcione sem acesso à internet; os
        demais gráficos reaproveitam essa mesma cópia já embutida na
        página, evitando repetir ~3MB de JS por gráfico.
        """
        if not graficos:
            return ""

        partes = ["<h2>Gráficos</h2>"]
        for indice, grafico in enumerate(graficos):
            figura = go.Figure(grafico.figura)
            partes.append(f"<h3>{html_stdlib.escape(grafico.titulo)}</h3>")
            if grafico.descricao:
                partes.append(f"<p>{html_stdlib.escape(grafico.descricao)}</p>")
            partes.append(
                figura.to_html(
                    full_html=False,
                    include_plotlyjs="inline" if indice == 0 else False,
                )
            )
        return "\n".join(partes)