"""Geração de relatório Markdown (engine.report) — único lugar que formata o relatório.

AVISO — placeholder de estrutura (fundação da Fase 1):

Segundo a Seção 3 do SDD v2.0, este módulo já existe na implementação
anterior do projeto e deve ser realocado para este caminho preservando
suas assinaturas públicas, sem reimplementação de lógica de negócio. Após
a relocação, este continua sendo o único lugar do projeto responsável por
formatar o relatório (não há `exporters/markdown_exporter.py` duplicado).

A relocação do código real ainda NÃO foi feita nesta conversa, porque o
conteúdo original (de `data_analysis_agent/report.py`, na estrutura
anterior) não está disponível neste contexto. Nenhuma lógica de geração
de relatório foi implementada ou reimplementada neste arquivo — ele
existe apenas para que a árvore de diretórios reflita o SDD.
"""
