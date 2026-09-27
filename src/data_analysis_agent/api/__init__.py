"""API REST (FastAPI) do AI Data Analysis Agent — Fase 5 do SDD v2.0.

Adaptador fino sobre o :class:`~data_analysis_agent.agent.agent.Agent`
(ponto de entrada único definido na Fase 4): os routers apenas convertem
requisições HTTP em chamadas a ``Agent.executar_pipeline`` /
``Agent.perguntar`` e serializam o resultado de volta — nenhuma regra de
negócio vive aqui.

Não existe ``api/schemas.py``: conforme a Seção 6 do SDD v2.0, os
endpoints reusam diretamente os modelos de domínio (Pydantic) definidos
em ``models/``.

Como a ``AnalysisEngine``/``Agent`` são stateless, a API também não
mantém estado entre requisições: cada chamada envia o dataset (upload) e
recebe de volta o resultado completo.
"""
