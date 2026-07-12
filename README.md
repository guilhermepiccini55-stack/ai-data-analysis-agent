# AI Data Analysis Agent

Agente de análise de dados com camada de IA, desenvolvido em fases conforme o
Software Design Document (SDD) v2.0.

## Status atual

**Fase 1 — Fundação do projeto.** Esta etapa contém apenas a estrutura de
diretórios, configuração, logging, exceções e os modelos de domínio (sem
lógica de negócio). Nenhuma funcionalidade de limpeza de dados, análise
estatística, visualização, relatório ou integração com IA foi implementada
ainda.

## Estrutura

```
src/data_analysis_agent/
├── config/        # Settings (pydantic-settings), constantes e logging
├── exceptions/     # Hierarquia centralizada de exceções
├── models/         # Modelos de domínio (Pydantic)
├── engine/         # AnalysisEngine (stateless) e módulos do pipeline
└── interfaces/
    └── streamlit_app/   # Interface Streamlit (Fase 1)
```

## Configuração do ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Executando a interface Streamlit

```bash
streamlit run src/data_analysis_agent/interfaces/streamlit_app/app.py
```

> Observação: a interface ainda exibe apenas uma página provisória — o
> dashboard interativo será implementado em uma etapa posterior da Fase 1.
