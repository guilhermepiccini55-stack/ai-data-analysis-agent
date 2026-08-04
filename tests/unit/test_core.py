import pandas as pd
import pytest

from data_analysis_agent.engine.core import AnalysisEngine
from data_analysis_agent.models.analysis_models import AnalysisResult, AnalysisSummary
from data_analysis_agent.models.data_models import CleaningReport
from data_analysis_agent.models.pipeline_models import ReportResult
from data_analysis_agent.models.report_models import ChartSpec, ReportMetadata, TipoGrafico


def test_analisar_retorna_analysis_summary() -> None:
    df = pd.DataFrame(
        {
            "idade": [20, 21, 22],
            "salario": [2000, 2500, 3000],
        }
    )

    engine = AnalysisEngine()

    resultado = engine.analisar(df)

    assert isinstance(resultado, AnalysisSummary)


def test_limpar_retorna_dataframe_e_cleaning_report() -> None:
    df = pd.DataFrame(
        {
            "nome": ["Ana", "Bruno"],
            "idade": [20, 21],
        }
    )

    engine = AnalysisEngine()

    dados_limpos, relatorio = engine.limpar(df)

    assert isinstance(dados_limpos, pd.DataFrame)
    assert isinstance(relatorio, CleaningReport)


def test_visualizar_delega_para_gerar_visualizacoes():
    dados = pd.DataFrame({"idade": [20, 30, 40]})
    resumo = AnalysisSummary()

    engine = AnalysisEngine()

    resultado = engine.visualizar(dados, resumo)

    assert isinstance(resultado, list)


# ---------------------------------------------------------------------------
# Fixtures compartilhadas para os testes de executar_pipeline() / gerar_relatorio()
# ---------------------------------------------------------------------------


@pytest.fixture
def cleaning_report_fake() -> CleaningReport:
    return CleaningReport(
        formato_original=(10, 3),
        formato_final=(9, 3),
        duplicatas_removidas=1,
        colunas_perfil=[],
        valores_imputados={},
        conversoes_tipo={},
    )


@pytest.fixture
def analysis_summary_fake() -> AnalysisSummary:
    return AnalysisSummary(estatisticas_descritivas={}, outliers=[], correlacoes=None)


@pytest.fixture
def chart_specs_fake() -> list[ChartSpec]:
    return [ChartSpec(titulo="Boxplot idade", tipo=TipoGrafico.BOXPLOT, figura={})]


@pytest.fixture
def report_result_fake() -> ReportResult:
    metadata = ReportMetadata(titulo="Relatório de teste", fonte_dados="fake")
    return ReportResult(metadata=metadata, markdown="# Relatório de teste")


@pytest.fixture
def analysis_result_parcial_fake(
    cleaning_report_fake: CleaningReport,
    analysis_summary_fake: AnalysisSummary,
    chart_specs_fake: list[ChartSpec],
) -> AnalysisResult:
    return AnalysisResult(
        fonte_dados="fake.csv",
        relatorio_limpeza=cleaning_report_fake,
        resultado_analise=analysis_summary_fake,
        graficos=chart_specs_fake,
        relatorio=None,
    )


def _aplicar_mocks_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    cleaning_report_fake: CleaningReport,
    analysis_summary_fake: AnalysisSummary,
    chart_specs_fake: list[ChartSpec],
    report_result_fake: ReportResult,
    registrar_chamadas: list[str] | None = None,
) -> None:
    """Aplica monkeypatch nas quatro funções internas usadas por executar_pipeline().

    Isola o teste da lógica real de limpeza/análise/visualização/relatório,
    testando apenas a orquestração feita pela engine.
    """

    def fake_limpar_dados(dados: pd.DataFrame):
        if registrar_chamadas is not None:
            registrar_chamadas.append("limpar")
        return dados, cleaning_report_fake

    def fake_analisar_dados(dados: pd.DataFrame):
        if registrar_chamadas is not None:
            registrar_chamadas.append("analisar")
        return analysis_summary_fake

    def fake_gerar_visualizacoes(dados: pd.DataFrame, resumo: AnalysisSummary):
        if registrar_chamadas is not None:
            registrar_chamadas.append("visualizar")
        return chart_specs_fake

    def fake_gerar_relatorio_markdown(resultado: AnalysisResult):
        if registrar_chamadas is not None:
            registrar_chamadas.append("gerar_relatorio")
        return report_result_fake

    monkeypatch.setattr(
        "data_analysis_agent.engine.core.limpar_dados", fake_limpar_dados
    )
    monkeypatch.setattr(
        "data_analysis_agent.engine.core.analisar_dados", fake_analisar_dados
    )
    monkeypatch.setattr(
        "data_analysis_agent.engine.core.gerar_visualizacoes", fake_gerar_visualizacoes
    )
    monkeypatch.setattr(
        "data_analysis_agent.engine.core.gerar_relatorio_markdown",
        fake_gerar_relatorio_markdown,
    )


# ---------------------------------------------------------------------------
# executar_pipeline()
# ---------------------------------------------------------------------------


class TestExecutarPipeline:
    def test_com_dataframe_direto_retorna_analysis_result_completo(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cleaning_report_fake: CleaningReport,
        analysis_summary_fake: AnalysisSummary,
        chart_specs_fake: list[ChartSpec],
        report_result_fake: ReportResult,
    ) -> None:
        _aplicar_mocks_pipeline(
            monkeypatch,
            cleaning_report_fake,
            analysis_summary_fake,
            chart_specs_fake,
            report_result_fake,
        )

        df_entrada = pd.DataFrame({"idade": [20, 30, 40]})
        engine = AnalysisEngine()

        resultado = engine.executar_pipeline(df_entrada)

        assert isinstance(resultado, AnalysisResult)
        assert resultado.fonte_dados == "DataFrame fornecido diretamente"
        assert resultado.relatorio_limpeza is cleaning_report_fake
        assert resultado.resultado_analise is analysis_summary_fake
        assert resultado.graficos == chart_specs_fake
        assert resultado.relatorio is report_result_fake

    def test_com_caminho_csv_carrega_arquivo_e_usa_caminho_como_fonte(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        cleaning_report_fake: CleaningReport,
        analysis_summary_fake: AnalysisSummary,
        chart_specs_fake: list[ChartSpec],
        report_result_fake: ReportResult,
    ) -> None:
        _aplicar_mocks_pipeline(
            monkeypatch,
            cleaning_report_fake,
            analysis_summary_fake,
            chart_specs_fake,
            report_result_fake,
        )

        csv_path = tmp_path / "dados.csv"
        pd.DataFrame({"idade": [20, 30, 40]}).to_csv(csv_path, index=False)

        engine = AnalysisEngine()
        resultado = engine.executar_pipeline(str(csv_path))

        assert resultado.fonte_dados == str(csv_path)
        assert resultado.relatorio is report_result_fake

    def test_com_caminho_path_object_tambem_funciona(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        cleaning_report_fake: CleaningReport,
        analysis_summary_fake: AnalysisSummary,
        chart_specs_fake: list[ChartSpec],
        report_result_fake: ReportResult,
    ) -> None:
        _aplicar_mocks_pipeline(
            monkeypatch,
            cleaning_report_fake,
            analysis_summary_fake,
            chart_specs_fake,
            report_result_fake,
        )

        csv_path = tmp_path / "dados.csv"
        pd.DataFrame({"idade": [20, 30, 40]}).to_csv(csv_path, index=False)

        engine = AnalysisEngine()
        resultado = engine.executar_pipeline(csv_path)

        assert resultado.fonte_dados == str(csv_path)

    def test_chama_etapas_na_ordem_correta(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cleaning_report_fake: CleaningReport,
        analysis_summary_fake: AnalysisSummary,
        chart_specs_fake: list[ChartSpec],
        report_result_fake: ReportResult,
    ) -> None:
        chamadas: list[str] = []
        _aplicar_mocks_pipeline(
            monkeypatch,
            cleaning_report_fake,
            analysis_summary_fake,
            chart_specs_fake,
            report_result_fake,
            registrar_chamadas=chamadas,
        )

        engine = AnalysisEngine()
        engine.executar_pipeline(pd.DataFrame({"a": [1, 2]}))

        assert chamadas == ["limpar", "analisar", "visualizar", "gerar_relatorio"]

    def test_duas_chamadas_seguidas_nao_interferem_entre_si(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cleaning_report_fake: CleaningReport,
        analysis_summary_fake: AnalysisSummary,
        chart_specs_fake: list[ChartSpec],
        report_result_fake: ReportResult,
    ) -> None:
        _aplicar_mocks_pipeline(
            monkeypatch,
            cleaning_report_fake,
            analysis_summary_fake,
            chart_specs_fake,
            report_result_fake,
        )

        engine = AnalysisEngine()

        resultado_a = engine.executar_pipeline(pd.DataFrame({"x": [1, 2]}))
        resultado_b = engine.executar_pipeline(pd.DataFrame({"y": [10, 20, 30]}))

        assert resultado_a is not resultado_b
        assert resultado_a.fonte_dados == resultado_b.fonte_dados == (
            "DataFrame fornecido diretamente"
        )
        # nenhum atributo de progresso foi acrescentado à instância
        assert vars(engine) == {"_dependencias": {}}


# ---------------------------------------------------------------------------
# gerar_relatorio()
# ---------------------------------------------------------------------------


class TestGerarRelatorio:
    def test_delega_para_gerar_relatorio_markdown(
        self,
        monkeypatch: pytest.MonkeyPatch,
        analysis_result_parcial_fake: AnalysisResult,
        report_result_fake: ReportResult,
    ) -> None:
        chamado_com: dict[str, AnalysisResult] = {}

        def fake_gerar_relatorio_markdown(resultado: AnalysisResult) -> ReportResult:
            chamado_com["resultado"] = resultado
            return report_result_fake

        monkeypatch.setattr(
            "data_analysis_agent.engine.core.gerar_relatorio_markdown",
            fake_gerar_relatorio_markdown,
        )

        engine = AnalysisEngine()
        resultado = engine.gerar_relatorio(analysis_result_parcial_fake)

        assert resultado is report_result_fake
        assert chamado_com["resultado"] is analysis_result_parcial_fake

    def test_retorna_report_result_real_sem_mock(
        self, analysis_result_parcial_fake: AnalysisResult
    ) -> None:
        """Teste de integração real (sem mock) com engine/report.py."""
        engine = AnalysisEngine()

        resultado = engine.gerar_relatorio(analysis_result_parcial_fake)

        assert isinstance(resultado, ReportResult)
        assert isinstance(resultado.markdown, str)
        assert len(resultado.markdown) > 0