"""Fixtures e utilitários compartilhados pelos testes de performance.

Diferente dos testes de integração (Seção "estratégia de testes" da SDD),
que usam um único dataset pequeno e realista para validar *comportamento*,
este módulo gera datasets sintéticos *escaláveis* para validar que o
pipeline não degrada de forma anômala (ex.: complexidade acidental O(n²))
à medida que o volume de dados cresce.

O dataset sintético preserva o mesmo perfil "sujo" usado em
``test_pipeline_completo.py`` (Fase de integração) — números em formato
brasileiro, datas ISO com ausentes, duplicatas e um outlier proposital —
mas com tamanho parametrizável via ``n_linhas``, para poder ser gerado em
1.000, 10.000 ou 50.000 linhas sem duplicar lógica de construção.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_SEED_PADRAO = 42


def gerar_dataset_sujo(n_linhas: int, seed: int = _SEED_PADRAO) -> pd.DataFrame:
    """Gera um DataFrame sintético "sujo" com ``n_linhas`` linhas.

    Características deliberadas (para forçar todas as etapas do pipeline):
        - ``Salário (R$)``: número em formato brasileiro (string), com um
          outlier proposital na última linha.
        - ``Data Contratação``: data ISO 8601 (string), ~2% dos valores
          ausentes.
        - ``Idade``: numérica, correlacionada com o salário (para gerar
          um par fortemente correlacionado e o respectivo gráfico de
          dispersão).
        - ``Departamento``: categórica, sem valores ausentes.
        - ``ID Cliente``: texto, não entra na análise estatística.
        - Aproximadamente 1% de linhas duplicadas (repete linhas do início
          do dataset), para exercitar a remoção de duplicatas em escala.

    Args:
        n_linhas: quantidade de linhas do dataset base (antes da
            duplicação de linhas para simular duplicatas).
        seed: semente do gerador aleatório, para reprodutibilidade.

    Returns:
        pd.DataFrame: dataset sintético, ainda não limpo.
    """
    rng = np.random.default_rng(seed)

    idades = rng.integers(20, 60, size=n_linhas).astype(float)
    salarios = 2000.0 + idades * 80.0 + rng.normal(0, 150, size=n_linhas)
    # Outlier proposital: um único valor muito acima da distribuição.
    salarios[-1] = salarios.max() * 5

    datas = pd.date_range("2018-01-01", periods=n_linhas, freq="D").astype(str).tolist()
    # ~2% de datas ausentes.
    n_ausentes = max(1, n_linhas // 50)
    indices_ausentes = rng.choice(n_linhas, size=n_ausentes, replace=False)
    for indice in indices_ausentes:
        datas[indice] = None

    departamentos = rng.choice(
        ["Vendas", "Engenharia", "Financeiro", "RH"], size=n_linhas
    )

    def formatar_numero_br(valor: float) -> str:
        inteiro, decimal = divmod(round(valor, 2), 1)
        return f"{inteiro:,.0f}".replace(",", ".") + f",{round(decimal * 100):02d}"

    df = pd.DataFrame(
        {
            "ID Cliente": [f"C{indice:06d}" for indice in range(n_linhas)],
            "Idade": idades,
            "Salário (R$)": [formatar_numero_br(valor) for valor in salarios],
            "Data Contratação": datas,
            "Departamento": departamentos,
        }
    )

    # ~1% de linhas duplicadas: repete as primeiras linhas ao final.
    n_duplicatas = max(1, n_linhas // 100)
    df_com_duplicatas = pd.concat([df, df.iloc[:n_duplicatas]], ignore_index=True)

    return df_com_duplicatas


@pytest.fixture
def gerador_dataset_sujo():
    """Expõe ``gerar_dataset_sujo`` como fixture, para uso direto nos testes."""
    return gerar_dataset_sujo