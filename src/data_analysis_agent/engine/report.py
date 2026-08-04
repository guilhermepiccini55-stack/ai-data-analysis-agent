"""Geração do relatório final em Markdown.

Conforme a Seção 7 do SDD, este é o único módulo responsável por
formatar o relatório em Markdown — nenhuma outra parte do sistema deve
duplicar essa lógica (a duplicação entre ``engine/report.py`` e um
possível ``exporters/markdown_exporter.py`` foi explicitamente fechada
como decisão arquitetural na v2.0).

Implementação mínima nesta fase: produz um resumo estruturado com as
contagens principais de cada etapa do pipeline (limpeza, análise,
visualização). Formatação mais rica (tabelas detalhadas, insights de
LLM) fica para fases futuras — não antecipar essa abstração agora
(YAGNI).
"""
from __future__ import annotations

from data_analysis_agent.models.analysis_models import AnalysisResult
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ReportMetadata


def gerar_relatorio_markdown(resultado: AnalysisResult) -> ReportResult:
    """Gera o relatório final em Markdown a partir de um ``AnalysisResult``.

    Args:
        resultado: resultado do pipeline já contendo limpeza, análise e
            visualizações (o campo ``relatorio`` é ignorado, caso já
            preenchido, já que é justamente o que esta função produz).

    Returns:
        ReportResult: metadados do relatório e seu conteúdo em Markdown.
    """
    secoes = ["Limpeza de Dados", "Análise Estatística", "Visualizações"]

    metadata = ReportMetadata(
        titulo=f"Relatório de Análise — {resultado.fonte_dados}",
        fonte_dados=resultado.fonte_dados,
        autor="AI Data Analysis Agent",
        secoes=secoes,
    )

    limpeza = resultado.relatorio_limpeza
    analise = resultado.resultado_analise
    graficos = resultado.graficos

    linhas: list[str] = []
    linhas.append(f"# {metadata.titulo}")
    linhas.append("")
    linhas.append(f"*Gerado em {metadata.gerado_em.strftime('%d/%m/%Y %H:%M')}*")
    linhas.append("")

    linhas.append("## Limpeza de Dados")
    linhas.append("")
    linhas.append(
        f"- Formato original: {limpeza.formato_original[0]} linhas × "
        f"{limpeza.formato_original[1]} colunas"
    )
    linhas.append(
        f"- Formato final: {limpeza.formato_final[0]} linhas × "
        f"{limpeza.formato_final[1]} colunas"
    )
    linhas.append(f"- Duplicatas removidas: {limpeza.duplicatas_removidas}")
    linhas.append(f"- Colunas com valores imputados: {len(limpeza.valores_imputados)}")
    linhas.append(f"- Colunas com conversão de tipo: {len(limpeza.conversoes_tipo)}")
    linhas.append("")

    linhas.append("## Análise Estatística")
    linhas.append("")
    linhas.append(
        f"- Colunas com estatísticas descritivas: {len(analise.estatisticas_descritivas)}"
    )
    linhas.append(f"- Colunas analisadas para outliers: {len(analise.outliers)}")
    total_outliers = sum(o.total_outliers for o in analise.outliers)
    linhas.append(f"- Total de outliers identificados: {total_outliers}")
    if analise.correlacoes is not None:
        n_pares = len(analise.correlacoes.pares_fortemente_correlacionados)
        linhas.append(f"- Pares fortemente correlacionados: {n_pares}")
    else:
        linhas.append("- Correlação: não aplicável (dados insuficientes)")
    linhas.append("")

    linhas.append("## Visualizações")
    linhas.append("")
    linhas.append(f"- Total de gráficos gerados: {len(graficos)}")
    for grafico in graficos:
        linhas.append(f"  - {grafico.titulo} ({grafico.tipo.value})")
    linhas.append("")

    markdown = "\n".join(linhas)

    return ReportResult(metadata=metadata, markdown=markdown)