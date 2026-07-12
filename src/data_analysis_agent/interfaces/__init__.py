"""Interfaces (adaptadores finos) do AI Data Analysis Agent.

Cada interface (Streamlit, futura API REST, futuro Agente) é responsável
por decidir se e como cacheia o `AnalysisResult` entre interações —
a `AnalysisEngine` não guarda esse estado, conforme a Seção 6 do SDD v2.0.
"""
