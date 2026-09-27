"""Teste de integração do pipeline completo (fim a fim).

Diferente dos testes unitários (que isolam cada etapa com fixtures
sintéticas mínimas), este módulo exercita `executar_pipeline()` de
ponta a ponta — limpeza, análise, visualização e relatório — sobre um
dataset único, realista e "sujo" (nomes de coluna com acentos/espaços,
números em formato brasileiro, datas ISO com valor ausente, uma
duplicata, uma coluna totalmente vazia, e um outlier proposital),
usando os componentes reais (sem mocks). Cobre também a entrada via
caminho de CSV e a delegação do `Agent` para a `AnalysisEngine`, sem
exigir LLM configurado — conforme a Seção 5 do SDD (a Engine é
stateless e `executar_pipeline` nunca depende de LLM).

Nome do arquivo: `test_pipeline_completo.py` (prefixo `test_` é exigido
pela configuração padrão do pytest para a coleta automática — o nome
"pipeline_completo.py", sem o prefixo, não seria coletado).
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_analysis_agent.agent.agent import Agent
from data_analysis_agent.config.settings import Settings
from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.models.analysis_models import AnalysisResult


@pytest.fixture
def dados_brutos_realistas() -> pd.DataFrame:
    """Dataset único que força todas as etapas do pipeline a agir.

    - Nomes de coluna com acentos/espaços/parênteses → normalização.
    - Uma linha duplicada (índices 4 e 5) → remoção de duplicatas.
    - Uma coluna 100% vazia → remoção de coluna vazia.
    - "Salário" em formato numérico brasileiro → conversão de tipo.
    - "Data Contratação" em ISO 8601, com 2 valores ausentes → conversão
      de tipo + imputação por interpolação.
    - "Idade" com um valor discrepante (90) → outlier detectado (IQR) e
      correlacionado com "Salário" (também com um valor alto) → boxplot
      + gráfico de dispersão gerados.
    - "ID Cliente" como texto (não numérico) → não entra na análise
      estatística/correlação, evitando correlação espúria com colunas
      monotônicas.
    """
    return pd.DataFrame(
        {
            "ID Cliente": ["C001", "C002", "C003", "C004", "C005", "C005", "C006"],
            "Idade": [25, 30, 28, 35, 32, 32, 90],
            "Salário (R$)": [
                "2.500,00",
                "3.000,00",
                "2.800,00",
                "3.500,00",
                "3.200,00",
                "3.200,00",
                "9.000,00",
            ],
            "Data Contratação": [
                "2020-01-15",
                "2019-05-10",
                "2021-03-20",
                "2018-07-01",
                None,
                None,
                "2015-02-11",
            ],
            "Coluna Vazia": [None] * 7,
        }
    )


def _validar_resultado_completo(resultado: AnalysisResult) -> None:
    """Asserções compartilhadas sobre o `AnalysisResult` produzido a
    partir de `dados_brutos_realistas`, reaproveitadas pelos testes que
    chegam a esse resultado por caminhos diferentes (Engine direta,
    Agent, ou a partir de um CSV em disco).
    """
    relatorio_limpeza = resultado.relatorio_limpeza
    resumo_analise = resultado.resultado_analise

    # --- Limpeza: estrutura ---
    assert relatorio_limpeza.formato_original == (7, 5)
    assert relatorio_limpeza.formato_final == (6, 4)  # 1 duplicata + 1 coluna vazia
    assert relatorio_limpeza.duplicatas_removidas == 1
    nomes_colunas_finais = {perfil.nome for perfil in relatorio_limpeza.colunas_perfil}
    assert nomes_colunas_finais == {"id_cliente", "idade", "salario_r", "data_contratacao"}

    # --- Limpeza: conversão de tipo e imputação ---
    assert relatorio_limpeza.conversoes_tipo["salario_r"] == "str -> float64"
    assert relatorio_limpeza.conversoes_tipo["data_contratacao"] == "str -> datetime64[us]"
    assert relatorio_limpeza.valores_imputados == {"data_contratacao": 1}

    # --- Análise: outliers e correlação ---
    outliers_por_coluna = {o.coluna: o.total_outliers for o in resumo_analise.outliers}
    assert outliers_por_coluna == {"idade": 1, "salario_r": 1}
    assert resumo_analise.correlacoes is not None
    assert resumo_analise.correlacoes.pares_fortemente_correlacionados == [
        ("idade", "salario_r", 1.0)
    ]

    # --- Visualização: um boxplot por coluna com outlier + uma dispersão pelo par correlacionado ---
    tipos_grafico = sorted(g.tipo.value for g in resultado.graficos)
    assert tipos_grafico == ["boxplot", "boxplot", "scatter"]
    titulos_grafico = {g.titulo for g in resultado.graficos}
    assert "idade vs salario_r (r=1.00)" in titulos_grafico

    # --- Relatório: reflete as mesmas contagens do restante do resultado ---
    assert resultado.relatorio is not None
    markdown = resultado.relatorio.markdown
    assert "Duplicatas removidas: 1" in markdown
    assert "Total de outliers identificados: 2" in markdown
    assert "Pares fortemente correlacionados: 1" in markdown
    assert "Total de gráficos gerados: 3" in markdown


class TestPipelineCompletoViaEngine:
    def test_pipeline_completo_produz_resultado_consistente(self, dados_brutos_realistas):
        engine = AnalysisEngine()

        resultado = engine.executar_pipeline(dados_brutos_realistas)

        assert resultado.fonte_dados == "DataFrame fornecido diretamente"
        _validar_resultado_completo(resultado)

    def test_pipeline_a_partir_de_caminho_csv(self, dados_brutos_realistas, tmp_path):
        caminho_csv = tmp_path / "clientes.csv"
        dados_brutos_realistas.to_csv(caminho_csv, index=False)
        engine = AnalysisEngine()

        resultado = engine.executar_pipeline(caminho_csv)

        assert resultado.fonte_dados == str(caminho_csv)
        _validar_resultado_completo(resultado)

    def test_duas_execucoes_consecutivas_com_datasets_diferentes_nao_interferem(
        self, dados_brutos_realistas
    ):
        engine = AnalysisEngine()
        outro_dataset = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        resultado_realista = engine.executar_pipeline(dados_brutos_realistas)
        resultado_simples = engine.executar_pipeline(outro_dataset)
        resultado_realista_de_novo = engine.executar_pipeline(dados_brutos_realistas)

        # A segunda execução (dataset diferente) não deixou resíduo na terceira.
        _validar_resultado_completo(resultado_realista_de_novo)
        assert resultado_simples.relatorio_limpeza.formato_original == (3, 2)
        assert all(o.total_outliers == 0 for o in resultado_simples.resultado_analise.outliers)
        # As duas execuções do mesmo dataset produzem o mesmo resultado.
        assert (
            resultado_realista.resultado_analise.model_dump()
            == resultado_realista_de_novo.resultado_analise.model_dump()
        )


class TestPipelineCompletoViaAgent:
    def test_agent_delega_para_engine_sem_exigir_llm(self, dados_brutos_realistas):
        settings_sem_api_key = Settings(anthropic_api_key=None)
        agent = Agent(AnalysisEngine(), settings=settings_sem_api_key)

        resultado = agent.executar_pipeline(dados_brutos_realistas)

        assert resultado.fonte_dados == "DataFrame fornecido diretamente"
        _validar_resultado_completo(resultado)

    def test_agent_executar_pipeline_nao_instancia_client_anthropic(
        self, dados_brutos_realistas, monkeypatch
    ):
        from unittest.mock import patch

        settings_sem_api_key = Settings(anthropic_api_key=None)
        agent = Agent(AnalysisEngine(), settings=settings_sem_api_key)

        with patch("data_analysis_agent.agent.agent.anthropic.Anthropic") as mock_anthropic:
            agent.executar_pipeline(dados_brutos_realistas)
            mock_anthropic.assert_not_called()