"""Testes de performance do pipeline (tests/performance/).

Objetivo: detectar regressões grosseiras de complexidade (ex.: um loop
O(n²) introduzido por engano) à medida que o volume de dados cresce —
não medir performance fina nem substituir profiling real. Por isso os
limites de tempo abaixo são deliberadamente generosos: o objetivo é
falhar quando algo está *ordens de grandeza* mais lento que o esperado,
não sinalizar variações normais de hardware/CI.

Não usa ``pytest-benchmark`` (nem qualquer outra dependência nova) —
medição via ``time.perf_counter``, consistente com a disciplina YAGNI da
SDD: uma dependência de benchmarking só se justificaria se este módulo
crescesse a ponto de precisar de estatísticas mais sofisticadas
(percentis, comparação histórica etc.), o que não é o caso aqui.

Os tamanhos de dataset (1.000 / 10.000 / 50.000 linhas) foram escolhidos
para cobrir uso típico (Streamlit interativo) até uso pesado (CSV grande
via API/Agente), sem deixar a suíte de performance lenta demais para
rodar rotineiramente — por isso este módulo fica separado da suíte
padrão via o marker ``performance`` (ver ``pyproject.toml``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import pytest

from data_analysis_agent.engine.core import AnalysisEngine
from tests.performance.conftest import gerar_dataset_sujo

pytestmark = pytest.mark.performance

# Limites superiores (em segundos) para o pipeline completo, por tamanho
# de dataset. Calibrados em sandbox (tempo observado: ~0.15s / ~0.35s /
# ~1.6s, respectivamente) com margem de ~8-10x para tolerar hardware mais
# lento sem virar teste flaky, mantendo ainda assim sensibilidade a uma
# regressão de complexidade real (ex.: um O(n²) faria o caso de 50.000
# linhas estourar o limite por larga margem, não por pouco).
_LIMITE_PIPELINE_COMPLETO: dict[int, float] = {
    1_000: 1.5,
    10_000: 3.5,
    50_000: 15.0,
}

# Limite por etapa granular, aplicado apenas ao maior dataset (50.000
# linhas) — é onde uma regressão de complexidade fica mais visível.
# Observado em sandbox: limpar ~1.56s (etapa dominante — conversão de
# tipos testa duas interpretações de data por coluna candidata),
# analisar ~0.02s, visualizar ~0.02s, gerar_relatorio ~0.00s.
_LIMITE_ETAPA_50K: dict[str, float] = {
    "limpar": 12.0,
    "analisar": 1.0,
    "visualizar": 1.0,
    "gerar_relatorio": 0.5,
}


@dataclass
class _Cronometro:
    """Pequeno utilitário para medir a duração de um bloco de código."""

    duracao_segundos: float = 0.0

    def __enter__(self) -> "_Cronometro":
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.duracao_segundos = time.perf_counter() - self._inicio


@pytest.fixture(scope="module")
def engine() -> AnalysisEngine:
    """Uma única instância de ``AnalysisEngine``, reutilizada entre os testes.

    Reutilizar a mesma instância entre execuções com tamanhos diferentes é
    proposital: reforça, sob volume, a garantia da Seção 5 do SDD de que a
    Engine é stateless — se houvesse vazamento de estado entre chamadas,
    seria mais provável de se manifestar como degradação de tempo aqui.
    """
    return AnalysisEngine()


@pytest.mark.parametrize("n_linhas", sorted(_LIMITE_PIPELINE_COMPLETO))
def test_pipeline_completo_escala_dentro_do_limite(
    engine: AnalysisEngine, n_linhas: int
) -> None:
    """``executar_pipeline()`` completa dentro do limite de tempo para cada tamanho."""
    dados = gerar_dataset_sujo(n_linhas)

    with _Cronometro() as cronometro:
        resultado = engine.executar_pipeline(dados)

    limite = _LIMITE_PIPELINE_COMPLETO[n_linhas]
    assert cronometro.duracao_segundos < limite, (
        f"Pipeline completo com {n_linhas} linhas levou "
        f"{cronometro.duracao_segundos:.2f}s (limite: {limite}s)."
    )
    # Confirma que o resultado é o esperado, não apenas rápido/vazio.
    assert resultado.relatorio_limpeza.formato_original[0] == n_linhas + max(1, n_linhas // 100)
    assert resultado.relatorio is not None


def test_etapas_granulares_50k_linhas_dentro_do_limite(engine: AnalysisEngine) -> None:
    """Cada etapa granular, isoladamente, completa dentro do limite em 50.000 linhas."""
    dados = gerar_dataset_sujo(50_000)

    with _Cronometro() as cronometro_limpar:
        dados_limpos, _relatorio_limpeza = engine.limpar(dados)
    assert cronometro_limpar.duracao_segundos < _LIMITE_ETAPA_50K["limpar"], (
        f"limpar() levou {cronometro_limpar.duracao_segundos:.2f}s "
        f"(limite: {_LIMITE_ETAPA_50K['limpar']}s)."
    )

    with _Cronometro() as cronometro_analisar:
        resumo_analise = engine.analisar(dados_limpos)
    assert cronometro_analisar.duracao_segundos < _LIMITE_ETAPA_50K["analisar"], (
        f"analisar() levou {cronometro_analisar.duracao_segundos:.2f}s "
        f"(limite: {_LIMITE_ETAPA_50K['analisar']}s)."
    )

    with _Cronometro() as cronometro_visualizar:
        graficos = engine.visualizar(dados_limpos, resumo_analise)
    assert cronometro_visualizar.duracao_segundos < _LIMITE_ETAPA_50K["visualizar"], (
        f"visualizar() levou {cronometro_visualizar.duracao_segundos:.2f}s "
        f"(limite: {_LIMITE_ETAPA_50K['visualizar']}s)."
    )

    from data_analysis_agent.models.analysis_models import AnalysisResult

    resultado_parcial = AnalysisResult(
        fonte_dados="dataset sintético de performance",
        relatorio_limpeza=_relatorio_limpeza,
        resultado_analise=resumo_analise,
        graficos=graficos,
        relatorio=None,
    )
    with _Cronometro() as cronometro_relatorio:
        engine.gerar_relatorio(resultado_parcial)
    assert cronometro_relatorio.duracao_segundos < _LIMITE_ETAPA_50K["gerar_relatorio"], (
        f"gerar_relatorio() levou {cronometro_relatorio.duracao_segundos:.2f}s "
        f"(limite: {_LIMITE_ETAPA_50K['gerar_relatorio']}s)."
    )


def test_execucoes_consecutivas_tamanhos_diferentes_nao_degradam(
    engine: AnalysisEngine,
) -> None:
    """Execuções alternando tamanhos pequenos/grandes não crescem de forma anômala.

    Reforça, sob volume, a garantia stateless da Seção 5: a mesma
    instância de ``AnalysisEngine`` processa datasets pequenos e grandes
    intercalados, e o tempo de cada execução pequena permanece
    consistente — sem sinal de acúmulo de estado/memória entre chamadas.
    """
    tempos_dataset_pequeno: list[float] = []

    for n_linhas in (1_000, 50_000, 1_000, 50_000, 1_000):
        dados = gerar_dataset_sujo(n_linhas)
        with _Cronometro() as cronometro:
            engine.executar_pipeline(dados)
        if n_linhas == 1_000:
            tempos_dataset_pequeno.append(cronometro.duracao_segundos)

    # As execuções de 1.000 linhas, intercaladas com as de 50.000, não
    # devem crescer de forma sustentada (tolerância generosa: até 4x a
    # primeira execução, para absorver variação normal do processo).
    primeira, *restantes = tempos_dataset_pequeno
    limite_degradacao = max(primeira * 4, 1.0)
    for tempo in restantes:
        assert tempo < limite_degradacao, (
            f"Execução de 1.000 linhas ficou {tempo:.3f}s após execuções "
            f"intercaladas de 50.000 linhas — primeira execução: "
            f"{primeira:.3f}s (limite: {limite_degradacao:.3f}s). Possível "
            f"vazamento de estado entre chamadas da Engine."
        )