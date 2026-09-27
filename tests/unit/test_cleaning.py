"""Testes unitários de engine.cleaning.

Testes black-box: verificam os resultados observáveis de
`limpar_dados` (DataFrame limpo + CleaningReport) e de
`normalizar_nomes_colunas`, sem depender da estratégia interna
(qual heurística de conversão foi usada, qual método de imputação
etc.) — apenas dos outcomes (contagens, dtypes, ausência de NaN).
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_analysis_agent.engine.cleaning import limpar_dados, normalizar_nomes_colunas
from data_analysis_agent.exceptions.errors import DataCleaningError
from data_analysis_agent.models.data_models import CleaningReport


class TestValidacaoEntrada:
    def test_entrada_nao_dataframe_levanta_data_cleaning_error(self):
        with pytest.raises(DataCleaningError):
            limpar_dados([{"a": 1}])

    def test_dataframe_vazio_nao_e_rejeitado(self):
        df_limpo, relatorio = limpar_dados(pd.DataFrame())

        assert isinstance(relatorio, CleaningReport)
        assert df_limpo.empty


class TestNormalizarNomesColunas:
    def test_normaliza_acentos_espacos_e_caracteres_especiais(self):
        df = pd.DataFrame({"Nome Completo": [1], "Salário (R$)!!": [2]})

        resultado = normalizar_nomes_colunas(df)

        assert list(resultado.columns) == ["nome_completo", "salario_r"]

    def test_colunas_duplicadas_apos_normalizacao_recebem_sufixo(self):
        df = pd.DataFrame({"Nome": [1], "nome": [2], "NOME": [3]})

        resultado = normalizar_nomes_colunas(df)

        assert list(resultado.columns) == ["nome", "nome_1", "nome_2"]

    def test_nome_vazio_apos_normalizacao_vira_coluna_sem_nome(self):
        df = pd.DataFrame({"!!!": [1]})

        resultado = normalizar_nomes_colunas(df)

        assert list(resultado.columns) == ["coluna_sem_nome"]

    def test_nao_modifica_dataframe_original(self):
        df = pd.DataFrame({"Nome Completo": [1]})

        normalizar_nomes_colunas(df)

        assert list(df.columns) == ["Nome Completo"]


class TestLimpezaEstrutural:
    def test_remove_colunas_totalmente_vazias(self):
        df = pd.DataFrame({"id": [1, 2], "vazio": [None, None]})

        df_limpo, _ = limpar_dados(df)

        assert "vazio" not in df_limpo.columns

    def test_remove_linhas_totalmente_vazias(self):
        df = pd.DataFrame({"id": [1, 2, None], "nome": ["Ana", "Bruno", None]})

        df_limpo, relatorio = limpar_dados(df)

        assert relatorio.formato_final[0] == 2

    def test_remove_duplicatas_e_registra_contagem_no_relatorio(self):
        df = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "nome": ["Ana", "Ana", "Bruno"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert relatorio.duplicatas_removidas == 1
        assert relatorio.formato_final[0] == 2

    def test_relatorio_registra_formato_original_e_final(self):
        df = pd.DataFrame({"id": [1, 2, 3], "valor": [10, 20, 30]})

        _, relatorio = limpar_dados(df)

        assert relatorio.formato_original == (3, 2)
        assert relatorio.formato_final == (3, 2)


class TestConversaoNumericaBrasileira:
    def test_converte_quando_taxa_de_sucesso_e_alta(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "valor": ["1.234,56", "2.000,00", "500,25", "10,00"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert pd.api.types.is_numeric_dtype(df_limpo["valor"])
        assert df_limpo["valor"].tolist() == [1234.56, 2000.0, 500.25, 10.0]
        assert "valor" in relatorio.conversoes_tipo

    def test_nao_converte_quando_taxa_de_sucesso_e_baixa(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "valor": ["1.234,56", "2.000,00", "500,25", "abc"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert not pd.api.types.is_numeric_dtype(df_limpo["valor"])
        assert "valor" not in relatorio.conversoes_tipo


class TestConversaoDatas:
    def test_converte_coluna_em_formato_iso(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "data": ["2024-01-15", "2024-02-20", "2024-03-10"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert pd.api.types.is_datetime64_any_dtype(df_limpo["data"])
        assert "data" in relatorio.conversoes_tipo

    def test_converte_data_nao_iso_quando_inequivoca(self):
        # Componente > 12 em pelo menos uma posição desfaz a ambiguidade
        # dayfirst/monthfirst — resultado não diverge entre as duas
        # interpretações, então a conversão é aceita.
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "data": ["13/05/2024", "14/06/2024", "15/07/2024"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert pd.api.types.is_datetime64_any_dtype(df_limpo["data"])
        assert "data" in relatorio.conversoes_tipo

    def test_recusa_conversao_quando_interpretacoes_divergem(self):
        # Todos os componentes <= 12: dayfirst=True e dayfirst=False
        # produzem datas válidas, porém diferentes entre si — a
        # conversão deve ser recusada e a coluna mantida como texto.
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "data": ["03/04/2024", "05/06/2024", "01/02/2024"],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert not pd.api.types.is_datetime64_any_dtype(df_limpo["data"])
        assert "data" not in relatorio.conversoes_tipo


class TestImputacaoValoresAusentes:
    def test_imputa_numericos_pela_mediana(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "idade": [20, None, 40, 30]})

        df_limpo, relatorio = limpar_dados(df)

        assert df_limpo["idade"].isna().sum() == 0
        assert df_limpo["idade"].tolist() == [20.0, 30.0, 40.0, 30.0]
        assert relatorio.valores_imputados["idade"] == 1

    def test_imputa_categoricos_pela_moda(self):
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5], "categoria": ["A", "A", "B", None, "A"]}
        )

        df_limpo, relatorio = limpar_dados(df)

        assert df_limpo["categoria"].isna().sum() == 0
        assert df_limpo["categoria"].tolist() == ["A", "A", "B", "A", "A"]
        assert relatorio.valores_imputados["categoria"] == 1

    def test_imputa_datas_por_interpolacao(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "data_evento": [
                    "2024-01-01",
                    "2024-01-02",
                    None,
                    "2024-01-04",
                    "2024-01-05",
                ],
            }
        )

        df_limpo, relatorio = limpar_dados(df)

        assert df_limpo["data_evento"].isna().sum() == 0
        assert df_limpo["data_evento"].tolist()[2] == pd.Timestamp("2024-01-03")
        assert relatorio.valores_imputados["data_evento"] == 1

    def test_nenhum_nan_restante_em_qualquer_coluna_apos_limpeza(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "idade": [20, None, 40, 30, 25],
                "categoria": ["A", "B", None, "A", "B"],
            }
        )

        df_limpo, _ = limpar_dados(df)

        assert df_limpo.isna().sum().sum() == 0


class TestPerfilColunas:
    def test_perfil_reflete_tipo_ausentes_unicos_e_amostra(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "categoria": ["A", "B", "A", None, "A"],
            }
        )

        _, relatorio = limpar_dados(df)
        perfil = next(p for p in relatorio.colunas_perfil if p.nome == "categoria")

        assert perfil.tipo_inferido == "categorical"
        assert perfil.total_valores == 5
        assert perfil.valores_ausentes == 0  # perfil é gerado pós-imputação
        assert perfil.valores_unicos == 2
        assert len(perfil.amostra_valores) == 5

    def test_perfil_gerado_para_todas_as_colunas(self):
        df = pd.DataFrame({"id": [1, 2], "nome": ["Ana", "Bruno"]})

        _, relatorio = limpar_dados(df)

        assert {p.nome for p in relatorio.colunas_perfil} == {"id", "nome"}