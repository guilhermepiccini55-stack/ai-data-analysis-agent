"""Limpeza de dados (engine.cleaning).

Implementa o contrato aprovado para a etapa de limpeza do pipeline
(Secao 5 do SDD v2.0). O modulo e stateless: nenhuma funcao guarda
estado entre chamadas - cada uma recebe o que precisa como parametro e
devolve um resultado completo.

API publica:
    - ``limpar_dados``: ponto de entrada unico, usado por
      ``AnalysisEngine.limpar()``.
    - ``normalizar_nomes_colunas``: mantida publica por ser util de forma
      isolada, fora do pipeline completo.

Todas as demais funcoes sao privadas e existem apenas para compor
``limpar_dados``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from typing import Any, Final

import numpy as np
import pandas as pd

from data_analysis_agent.exceptions.errors import DataCleaningError
from data_analysis_agent.models.data_models import CleaningReport, ColumnProfile

logger = logging.getLogger(__name__)

_LIMIAR_CONVERSAO_SEGURA: Final[float] = 0.9
"""Proporcao minima de valores convertidos com sucesso para que uma
conversao de tipo (numerica ou de data) seja aceita."""

_TAMANHO_AMOSTRA_PADRAO: Final[int] = 5
"""Quantidade padrao de valores extraidos como amostra em ColumnProfile."""

_PADRAO_ISO_DATA: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)
"""Reconhece datas/horas em formato ISO 8601 (ex.: 2024-01-15, 2024-01-15T10:30:00)."""

_PADRAO_NUMERO_BR: Final[re.Pattern[str]] = re.compile(
    r"^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+(,\d+)?$"
)
"""Reconhece numeros em formato brasileiro (milhar '.', decimal ',')."""


def normalizar_nomes_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza os nomes das colunas para snake_case, sem acentos ou caracteres especiais.

    Nomes que colidirem apos a normalizacao recebem um sufixo numerico
    incremental para preservar a unicidade das colunas.

    Args:
        df: DataFrame cujas colunas serao renomeadas.

    Returns:
        pd.DataFrame: copia de ``df`` com os nomes de colunas normalizados.
    """
    df_normalizado = df.copy()

    nomes_normalizados = [_normalizar_nome_coluna_individual(str(coluna)) for coluna in df.columns]

    contagem: dict[str, int] = {}
    nomes_finais: list[str] = []
    for nome in nomes_normalizados:
        contagem[nome] = contagem.get(nome, 0) + 1
        if contagem[nome] == 1:
            nomes_finais.append(nome)
        else:
            nomes_finais.append(f"{nome}_{contagem[nome] - 1}")

    df_normalizado.columns = nomes_finais
    return df_normalizado


def limpar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Executa o pipeline completo de limpeza de dados.

    Ordem de execucao:
        1. Validacao do tipo de entrada.
        2. Normalizacao dos nomes de colunas.
        3. Remocao de colunas totalmente vazias.
        4. Remocao de linhas totalmente vazias.
        5. Remocao de duplicatas.
        6. Captura dos dtypes originais (pos-limpeza estrutural, pre-conversao).
        7. Conversao de tipos (numeros em formato brasileiro, depois datas).
        8. Inferencia do tipo semantico de cada coluna.
        9. Tratamento de valores ausentes.
        10. Geracao do perfil de colunas (sobre os dados ja imputados).

    Args:
        df: DataFrame de entrada a ser limpo.

    Returns:
        tuple[pd.DataFrame, CleaningReport]: par com o DataFrame limpo e o
        relatorio descrevendo todas as operacoes realizadas.

    Raises:
        DataCleaningError: se ``df`` nao for uma instancia de ``pd.DataFrame``.
    """
    _validar_tipo_dataframe(df)

    formato_original = df.shape

    df_limpo = normalizar_nomes_colunas(df)
    df_limpo = _remover_colunas_vazias(df_limpo)
    df_limpo = _remover_linhas_vazias(df_limpo)
    df_limpo, duplicatas_removidas = _remover_duplicatas(df_limpo)

    tipos_originais = {coluna: str(df_limpo[coluna].dtype) for coluna in df_limpo.columns}

    df_limpo, conversoes_tipo = _converter_tipos(df_limpo)

    tipos_inferidos = _inferir_tipos_colunas(df_limpo)

    df_limpo, valores_imputados = _tratar_valores_ausentes(df_limpo, tipos_inferidos)

    colunas_perfil = _gerar_perfil_colunas(df_limpo, tipos_originais, tipos_inferidos)

    relatorio = CleaningReport(
        formato_original=formato_original,
        formato_final=df_limpo.shape,
        duplicatas_removidas=duplicatas_removidas,
        colunas_perfil=colunas_perfil,
        valores_imputados=valores_imputados,
        conversoes_tipo=conversoes_tipo,
    )

    return df_limpo, relatorio


def _validar_tipo_dataframe(df: pd.DataFrame) -> None:
    """Valida que a entrada e uma instancia de ``pd.DataFrame``.

    DataFrames vazios (sem linhas e/ou sem colunas) nao sao rejeitados -
    apenas o tipo da entrada e validado.

    Args:
        df: objeto a ser validado.

    Raises:
        DataCleaningError: se ``df`` nao for uma instancia de ``pd.DataFrame``.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataCleaningError(
            f"Entrada invalida para limpeza de dados: esperado pd.DataFrame, "
            f"recebido {type(df).__name__}."
        )


def _normalizar_nome_coluna_individual(nome: str) -> str:
    """Normaliza um unico nome de coluna para snake_case, sem acentos.

    Args:
        nome: nome original da coluna.

    Returns:
        str: nome normalizado. Se o resultado ficar vazio, devolve
        ``"coluna_sem_nome"``.
    """
    nome_sem_acentos = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    nome_minusculo = nome_sem_acentos.strip().lower()
    nome_com_underscores = re.sub(r"[^a-z0-9]+", "_", nome_minusculo)
    nome_normalizado = re.sub(r"_+", "_", nome_com_underscores).strip("_")
    return nome_normalizado or "coluna_sem_nome"


def _remover_colunas_vazias(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas cujos valores sao 100% nulos.

    Args:
        df: DataFrame de entrada.

    Returns:
        pd.DataFrame: DataFrame sem as colunas totalmente vazias.
    """
    return df.dropna(axis=1, how="all")


def _remover_linhas_vazias(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas cujos valores sao 100% nulos.

    Args:
        df: DataFrame de entrada.

    Returns:
        pd.DataFrame: DataFrame sem as linhas totalmente vazias, com o
        indice reiniciado.
    """
    return df.dropna(axis=0, how="all").reset_index(drop=True)


def _remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove linhas duplicadas.

    Args:
        df: DataFrame de entrada.

    Returns:
        tuple[pd.DataFrame, int]: DataFrame sem duplicatas (indice
        reiniciado) e a quantidade de linhas removidas.
    """
    linhas_antes = len(df)
    df_sem_duplicatas = df.drop_duplicates().reset_index(drop=True)
    duplicatas_removidas = linhas_antes - len(df_sem_duplicatas)
    return df_sem_duplicatas, duplicatas_removidas


def _calcular_taxa_conversao_valida(original: pd.Series, convertido: pd.Series) -> float:
    """Calcula a proporcao de valores convertidos com sucesso.

    Args:
        original: serie original, sem os valores nulos que ja existiam
            antes da tentativa de conversao.
        convertido: serie resultante da tentativa de conversao, alinhada
            por indice com ``original``.

    Returns:
        float: proporcao (entre 0.0 e 1.0) de valores nao nulos em
        ``convertido`` em relacao ao total de ``original``. Devolve 0.0
        se ``original`` estiver vazia.
    """
    total = len(original)
    if total == 0:
        return 0.0
    sucesso = int(convertido.notna().sum())
    return sucesso / total


def _tentar_conversao_numerica_brasileira(serie: pd.Series) -> pd.Series | None:
    """Tenta converter uma serie textual para numerico em formato brasileiro.

    Substitui '.' (separador de milhar) por nada e ',' (separador
    decimal) por '.', entao tenta a conversao numerica. A conversao so e
    aceita se a taxa de sucesso atingir ``_LIMIAR_CONVERSAO_SEGURA``.

    Args:
        serie: serie textual candidata a conversao numerica.

    Returns:
        pd.Series | None: serie numerica convertida, ou ``None`` se a
        serie nao tiver valores nao nulos, nao contiver nenhum candidato
        plausivel a numero brasileiro, ou a taxa de conversao ficar
        abaixo do limiar.
    """
    valores_nao_nulos = serie.dropna()
    if valores_nao_nulos.empty:
        return None

    valores_texto = valores_nao_nulos.astype(str).str.strip()
    candidatos = valores_texto[valores_texto.str.match(_PADRAO_NUMERO_BR)]
    if candidatos.empty:
        return None

    valores_normalizados = valores_texto.str.replace(".", "", regex=False).str.replace(
        ",", ".", regex=False
    )
    convertido_nao_nulos = pd.to_numeric(valores_normalizados, errors="coerce")

    taxa = _calcular_taxa_conversao_valida(valores_nao_nulos, convertido_nao_nulos)
    if taxa < _LIMIAR_CONVERSAO_SEGURA:
        return None

    texto_completo = serie.astype(str).str.strip()
    texto_normalizado = texto_completo.str.replace(".", "", regex=False).str.replace(
        ",", ".", regex=False
    )
    serie_final = pd.to_numeric(texto_normalizado, errors="coerce")
    serie_final = serie_final.where(serie.notna(), other=np.nan)
    return serie_final


def _eh_formato_iso(serie: pd.Series) -> bool:
    """Detecta se todos os valores nao nulos da serie estao em formato ISO 8601.

    Usado para evitar aplicar a heuristica de ambiguidade ``dayfirst`` a
    datas que ja estao em um formato inequivoco.

    Args:
        serie: serie textual candidata a data.

    Returns:
        bool: ``True`` se a serie nao estiver vazia e todos os valores
        nao nulos casarem com o padrao ISO 8601; ``False`` caso contrario.
    """
    valores_nao_nulos = serie.dropna().astype(str).str.strip()
    if valores_nao_nulos.empty:
        return False
    correspondencias = valores_nao_nulos.str.match(_PADRAO_ISO_DATA)
    return bool(correspondencias.all())


def _tentar_conversao_data(serie: pd.Series, dayfirst: bool) -> pd.Series | None:
    """Tenta converter uma serie textual para data com uma interpretacao fixa de ``dayfirst``.

    Nao utiliza ``format="mixed"`` para evitar a regressao de desempenho
    de inferencia de formato indiscriminada por valor.

    Args:
        serie: serie textual candidata a conversao de data.
        dayfirst: se ``True``, interpreta o primeiro componente numerico
            ambiguo como dia; se ``False``, como mes.

    Returns:
        pd.Series | None: serie de datas convertida, ou ``None`` se a
        serie nao tiver valores nao nulos ou a taxa de conversao ficar
        abaixo de ``_LIMIAR_CONVERSAO_SEGURA``.
    """
    valores_nao_nulos = serie.dropna()
    if valores_nao_nulos.empty:
        return None

    with warnings.catch_warnings():
        # O pandas emite UserWarning ao inferir formato por valor (fallback
        # para dateutil) ou ao usar o dayfirst padrão. Isso é esperado aqui,
        # pois a ambiguidade já é resolvida explicitamente pelo chamador
        # (_converter_datas), então o aviso é apenas ruído.
        warnings.simplefilter("ignore", UserWarning)
        convertido = pd.to_datetime(serie, dayfirst=dayfirst, errors="coerce")

    convertido_nao_nulos = convertido.loc[valores_nao_nulos.index]

    taxa = _calcular_taxa_conversao_valida(valores_nao_nulos, convertido_nao_nulos)
    if taxa < _LIMIAR_CONVERSAO_SEGURA:
        return None

    return convertido


def _series_datas_divergem(serie_a: pd.Series, serie_b: pd.Series) -> bool:
    """Compara duas interpretacoes de conversao de data, linha a linha.

    Args:
        serie_a: serie de datas resultante de uma interpretacao (ex.: dayfirst=True).
        serie_b: serie de datas resultante da outra interpretacao (ex.: dayfirst=False).

    Returns:
        bool: ``True`` se houver ao menos uma linha em que ambas as
        interpretacoes produziram um valor valido, mas divergente entre
        si; ``False`` caso contrario (inclusive se nao houver nenhuma
        linha em que ambas sejam validas simultaneamente).
    """
    ambas_validas = serie_a.notna() & serie_b.notna()
    if not ambas_validas.any():
        return False
    return bool((serie_a[ambas_validas] != serie_b[ambas_validas]).any())


def _converter_numeros_formato_brasileiro(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Converte colunas textuais em formato numerico brasileiro para numerico.

    Percorre apenas colunas de dtype textual (``object`` ou ``string``).
    Colunas onde a conversao nao atinge ``_LIMIAR_CONVERSAO_SEGURA``
    permanecem inalteradas.

    Args:
        df: DataFrame de entrada.

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: DataFrame com as colunas
        convertidas onde aplicavel, e um dicionario
        ``{coluna: "tipo_original -> tipo_final"}`` com as conversoes
        efetivamente aplicadas.
    """
    df_convertido = df.copy()
    conversoes: dict[str, str] = {}

    for coluna in df_convertido.columns:
        serie = df_convertido[coluna]
        if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
            continue

        tipo_original = str(serie.dtype)
        serie_convertida = _tentar_conversao_numerica_brasileira(serie)
        if serie_convertida is not None:
            df_convertido[coluna] = serie_convertida
            conversoes[coluna] = f"{tipo_original} -> {serie_convertida.dtype}"

    return df_convertido, conversoes


def _converter_datas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Converte colunas textuais candidatas a data/hora.

    Colunas ja em formato ISO 8601 sao convertidas diretamente, sem
    aplicar a heuristica de ambiguidade ``dayfirst``. Para as demais,
    testa as duas interpretacoes (dayfirst=True e dayfirst=False);
    se ambas atingirem o limiar de seguranca e divergirem linha a linha,
    a conversao e recusada e um aviso e registrado no log. Se apenas uma
    interpretacao atingir o limiar, essa e usada.

    Args:
        df: DataFrame de entrada (ja processado por
            ``_converter_numeros_formato_brasileiro``).

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: DataFrame com as colunas de
        data convertidas onde aplicavel, e um dicionario
        ``{coluna: "tipo_original -> tipo_final"}`` com as conversoes
        efetivamente aplicadas.
    """
    df_convertido = df.copy()
    conversoes: dict[str, str] = {}

    for coluna in df_convertido.columns:
        serie = df_convertido[coluna]
        if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
            continue

        tipo_original = str(serie.dtype)

        if _eh_formato_iso(serie):
            serie_convertida = _tentar_conversao_data(serie, dayfirst=False)
            if serie_convertida is not None:
                df_convertido[coluna] = serie_convertida
                conversoes[coluna] = f"{tipo_original} -> {serie_convertida.dtype}"
            continue

        serie_dayfirst = _tentar_conversao_data(serie, dayfirst=True)
        serie_monthfirst = _tentar_conversao_data(serie, dayfirst=False)

        if serie_dayfirst is not None and serie_monthfirst is not None:
            if _series_datas_divergem(serie_dayfirst, serie_monthfirst):
                logger.warning(
                    "Conversao de data ambigua na coluna '%s': as interpretacoes "
                    "dayfirst=True e dayfirst=False divergem entre si. "
                    "Conversao recusada; coluna mantida como texto.",
                    coluna,
                )
                continue
            df_convertido[coluna] = serie_dayfirst
            conversoes[coluna] = f"{tipo_original} -> {serie_dayfirst.dtype}"
        elif serie_dayfirst is not None:
            df_convertido[coluna] = serie_dayfirst
            conversoes[coluna] = f"{tipo_original} -> {serie_dayfirst.dtype}"
        elif serie_monthfirst is not None:
            df_convertido[coluna] = serie_monthfirst
            conversoes[coluna] = f"{tipo_original} -> {serie_monthfirst.dtype}"

    return df_convertido, conversoes


def _converter_tipos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Consolida a conversao de tipos do DataFrame.

    Executa primeiro a conversao numerica em formato brasileiro e, sobre
    o resultado, a conversao de datas - nessa ordem, para evitar que o
    parser de datas interprete precocemente numeros em formato brasileiro
    (ex.: precos como "1.234,56") como datas.

    Args:
        df: DataFrame de entrada (ja sem duplicatas e linhas/colunas vazias).

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: DataFrame com todas as
        conversoes de tipo aplicadas, e o dicionario combinado
        ``{coluna: "tipo_original -> tipo_final"}`` das duas etapas.
    """
    df_numeros, conversoes_numeros = _converter_numeros_formato_brasileiro(df)
    df_convertido, conversoes_datas = _converter_datas(df_numeros)
    conversoes_tipo = {**conversoes_numeros, **conversoes_datas}
    return df_convertido, conversoes_tipo


def _classificar_tipo_coluna(serie: pd.Series) -> str:
    """Classifica uma serie ja convertida em uma categoria semantica.

    Args:
        serie: serie (ja processada por ``_converter_tipos``) a ser classificada.

    Returns:
        str: uma de ``"date"``, ``"boolean"``, ``"numeric"`` ou ``"categorical"``.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "date"
    if pd.api.types.is_bool_dtype(serie):
        return "boolean"
    if pd.api.types.is_numeric_dtype(serie):
        return "numeric"
    return "categorical"


def _inferir_tipos_colunas(df: pd.DataFrame) -> dict[str, str]:
    """Infere o tipo semantico de cada coluna do DataFrame.

    Args:
        df: DataFrame ja processado por ``_converter_tipos``.

    Returns:
        dict[str, str]: mapa ``{coluna: tipo_inferido}``.
    """
    return {coluna: _classificar_tipo_coluna(df[coluna]) for coluna in df.columns}


def _imputar_valores_numericos(serie: pd.Series) -> pd.Series:
    """Imputa valores ausentes de uma coluna numerica pela mediana.

    Args:
        serie: serie numerica com valores ausentes.

    Returns:
        pd.Series: serie com os valores ausentes preenchidos pela
        mediana dos valores existentes. Se nao houver nenhum valor
        valido para calcular a mediana, a serie e devolvida inalterada.
    """
    mediana = serie.median()
    if pd.isna(mediana):
        return serie
    return serie.fillna(mediana)


def _imputar_valores_categoricos(serie: pd.Series) -> pd.Series:
    """Imputa valores ausentes de uma coluna categorica pela moda.

    Args:
        serie: serie categorica com valores ausentes.

    Returns:
        pd.Series: serie com os valores ausentes preenchidos pelo valor
        mais frequente. Se nao houver nenhum valor valido, a serie e
        devolvida inalterada.
    """
    modas = serie.mode(dropna=True)
    if modas.empty:
        return serie
    return serie.fillna(modas.iloc[0])


def _imputar_valores_data(serie: pd.Series) -> pd.Series:
    """Imputa valores ausentes de uma coluna de data por interpolacao linear.

    A interpolacao de datas nao e suportada diretamente pelo pandas, entao
    a serie e convertida para timestamps inteiros (respeitando a
    resolucao original - ns, us, ms ou s - para compatibilidade com
    pandas 3.x), interpolada nesse espaco numerico e entao reconvertida
    para data.

    Args:
        serie: serie de datas (``datetime64``) com valores ausentes.

    Returns:
        pd.Series: serie com os valores ausentes preenchidos por
        interpolacao linear. Se houver menos de dois valores validos
        para interpolar, recorre a moda; se nao houver nenhum valor
        valido, a serie e devolvida inalterada.
    """
    valores_validos = int(serie.notna().sum())
    if valores_validos < 2:
        modas = serie.mode(dropna=True)
        if modas.empty:
            return serie
        return serie.fillna(modas.iloc[0])

    unidade = np.datetime_data(serie.to_numpy().dtype)[0]

    timestamps = serie.astype("int64").astype("float64")
    timestamps[serie.isna()] = np.nan

    timestamps_interpolados = timestamps.interpolate(method="linear", limit_direction="both")
    valores_inteiros = timestamps_interpolados.round().astype("int64")
    serie_interpolada = pd.to_datetime(valores_inteiros, unit=unidade)
    return serie_interpolada


def _tratar_valores_ausentes(
    df: pd.DataFrame, tipos_inferidos: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Imputa valores ausentes em todas as colunas do DataFrame.

    A estrategia de imputacao depende do tipo semantico inferido de cada
    coluna: mediana para numericas, interpolacao para datas, e moda para
    as demais (categoricas e booleanas).

    Args:
        df: DataFrame ja convertido, antes da imputacao.
        tipos_inferidos: mapa ``{coluna: tipo_inferido}`` produzido por
            ``_inferir_tipos_colunas``.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: DataFrame com os valores
        ausentes imputados, e um dicionario ``{coluna: quantidade_imputada}``
        com a quantidade de valores efetivamente preenchidos por coluna.
    """
    df_tratado = df.copy()
    valores_imputados: dict[str, int] = {}

    for coluna in df_tratado.columns:
        serie = df_tratado[coluna]
        ausentes_antes = int(serie.isna().sum())
        if ausentes_antes == 0:
            continue

        tipo = tipos_inferidos.get(coluna, "categorical")
        if tipo == "numeric":
            serie_imputada = _imputar_valores_numericos(serie)
        elif tipo == "date":
            serie_imputada = _imputar_valores_data(serie)
        else:
            serie_imputada = _imputar_valores_categoricos(serie)

        df_tratado[coluna] = serie_imputada
        ausentes_depois = int(serie_imputada.isna().sum())
        valores_imputados[coluna] = ausentes_antes - ausentes_depois

    return df_tratado, valores_imputados


def _gerar_amostra_valores(serie: pd.Series, tamanho: int = _TAMANHO_AMOSTRA_PADRAO) -> list[Any]:
    """Extrai uma amostra de valores nao nulos de uma serie.

    Args:
        serie: serie da qual extrair a amostra.
        tamanho: quantidade maxima de valores na amostra.

    Returns:
        list[Any]: ate ``tamanho`` valores nao nulos, na ordem em que
        aparecem na serie. Lista vazia se nao houver valores nao nulos.
    """
    valores_nao_nulos = serie.dropna()
    if valores_nao_nulos.empty:
        return []
    return valores_nao_nulos.head(tamanho).tolist()


def _gerar_perfil_colunas(
    df: pd.DataFrame, tipos_originais: dict[str, str], tipos_inferidos: dict[str, str]
) -> list[ColumnProfile]:
    """Gera o perfil descritivo de cada coluna do DataFrame.

    Executada sobre o DataFrame ja imputado, de modo que o perfil
    reflita o estado final dos dados (apos limpeza, conversao e
    tratamento de valores ausentes).

    Args:
        df: DataFrame ja limpo, convertido e imputado.
        tipos_originais: mapa ``{coluna: dtype_original}``, capturado
            apos a limpeza estrutural e antes das conversoes de tipo.
        tipos_inferidos: mapa ``{coluna: tipo_inferido}`` produzido por
            ``_inferir_tipos_colunas``.

    Returns:
        list[ColumnProfile]: um perfil por coluna do DataFrame.
    """
    perfis: list[ColumnProfile] = []
    total_linhas = len(df)

    for coluna in df.columns:
        serie = df[coluna]
        valores_ausentes = int(serie.isna().sum())
        percentual_ausente = (valores_ausentes / total_linhas * 100) if total_linhas > 0 else 0.0

        perfil = ColumnProfile(
            nome=coluna,
            tipo_inferido=tipos_inferidos.get(coluna, "categorical"),
            tipo_original=tipos_originais.get(coluna, str(serie.dtype)),
            total_valores=total_linhas,
            valores_ausentes=valores_ausentes,
            percentual_ausente=round(percentual_ausente, 2),
            valores_unicos=int(serie.nunique(dropna=True)),
            amostra_valores=_gerar_amostra_valores(serie),
        )
        perfis.append(perfil)

    return perfis