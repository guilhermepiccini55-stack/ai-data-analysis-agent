# AI Data Analysis Agent

## Visão geral

O **AI Data Analysis Agent** é um agente de análise de dados com camada de IA, desenvolvido conforme o Software Design Document (SDD) v2.0, com um pipeline **stateless** para análise automatizada de arquivos CSV.

A camada determinística de análise (limpeza, análise estatística, visualizações e geração de relatório) funciona de forma independente e **não depende de nenhum provider de LLM**. Já o Chat com tool-calling e a geração de insights via IA dependem da configuração de uma chave de API da Anthropic — sem ela, essas funcionalidades ficam indisponíveis, mas o restante do pipeline continua operando normalmente.

## Funcionalidades

- Carregamento de arquivos CSV.
- Limpeza de dados (tratamento de duplicatas, perfil de colunas etc.).
- Análise estatística dos dados.
- Identificação e análise de correlações.
- Geração de visualizações (gráficos).
- Geração de relatório em Markdown.
- Exportação do relatório em Markdown.
- Exportação do relatório em HTML (autocontido/offline).
- Exportação do relatório em PDF.
- `Agent` com tool-calling para responder perguntas em linguagem natural sobre um dataset.
- Integração com a API da Anthropic para geração de insights e chat.
- Interface Streamlit para uso interativo.
- API REST via FastAPI.
- Execução em Docker.
- Orquestração dos serviços via Docker Compose.
- Testes unitários.
- Testes de integração.
- Testes de performance (isolados da suíte padrão).
- Integração contínua (CI) via GitHub Actions.

## Arquitetura

- **`AnalysisEngine`** (`engine/core.py`): motor de orquestração do pipeline, stateless — nenhum método guarda estado de uma análise específica entre chamadas.
- **`Agent`** (`agent/`): camada de orquestração central, usada tanto pela interface Streamlit quanto pela API. Concentra a lógica de tool-calling e a checagem da configuração do provider de LLM.
- **`engine/report.py`**: fonte central de geração do relatório em Markdown — os exporters (Markdown, HTML, PDF) reutilizam essa mesma representação, sem duplicar a lógica de relatório.
- **`ChartSpec`**: modelo que armazena a representação serializável de cada gráfico gerado, permitindo que ela seja reaproveitada entre engine, relatório e exporters.
- **API (`api/`)**: reutiliza diretamente os modelos de domínio (Pydantic) do restante do projeto; não existe um `schemas.py` paralelo.
- **Exporters (`exporters/`)**: reutilizam a cadeia de geração de relatório já existente (Markdown como base), sem lógica de conteúdo duplicada.
- **Streamlit (`interfaces/streamlit_app/`)**: responsável pelo estado de sessão da interface (upload atual, resultado da análise em memória etc.).
- **API**: não mantém estado persistente entre requisições — cada chamada recebe os dados necessários e devolve um resultado completo e autossuficiente.
- **Injeção de dependências**: simples, por parâmetro/construtor (ex.: `AnalysisEngine(llm_provider=...)`), sem uso de framework de DI.

## Estrutura do projeto

```
ai-data-analysis-agent/
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── .env.example
├── README.md
├── src/
│   └── data_analysis_agent/
│       ├── config/
│       ├── exceptions/
│       ├── models/
│       ├── engine/
│       ├── llm/
│       ├── agent/
│       ├── api/
│       │   └── routers/
│       ├── exporters/
│       └── interfaces/
│           └── streamlit_app/
└── tests/
    ├── unit/
    ├── integration/
    └── performance/
```

## Requisitos

- **Python 3.14** — versão utilizada e validada pelo projeto (incluindo dentro da imagem Docker, `python:3.14-slim`). Não é recomendado usar versões anteriores.

## Configuração

### Ambiente virtual

```bash
python -m venv .venv
```

Ativação no Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Ativação em sistemas Unix-like:

```bash
source .venv/bin/activate
```

### Instalação

```bash
pip install -e ".[dev]"
```

### Variáveis de ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Principais variáveis (prefixo `DAA_`):

| Variável | Descrição |
|---|---|
| `DAA_DEBUG` | Habilita o modo debug da aplicação. |
| `DAA_NIVEL_LOG` | Nível mínimo de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `DAA_DIRETORIO_SAIDA` | Diretório padrão para arquivos de saída da aplicação. |
| `DAA_DIRETORIO_DADOS` | Diretório padrão para datasets de entrada. |
| `DAA_ANTHROPIC_API_KEY` | Chave de API da Anthropic, usada pelo provider de LLM para gerar insights e habilitar o Chat. |
| `DAA_ANTHROPIC_MODEL` | Modelo Claude usado pelo provider ao gerar insights. |

> **Importante:** `DAA_ANTHROPIC_API_KEY` é uma credencial sensível e **não deve ser commitada** em nenhuma hipótese. Mantenha-a apenas no seu `.env` local.

## Execução local

Para rodar a interface Streamlit diretamente:

```bash
streamlit run src/data_analysis_agent/interfaces/streamlit_app/app.py
```

Fluxo esperado: carregue um arquivo CSV pela interface, execute a análise, visualize os resultados (limpeza, estatísticas, gráficos), gere o relatório e, se desejar, exporte-o em Markdown, HTML ou PDF. Se `DAA_ANTHROPIC_API_KEY` estiver configurada, é possível também interagir com o Chat sobre o dataset carregado.

## Execução com Docker

O projeto usa uma única imagem Docker, compartilhada entre a API e a interface Streamlit — a diferença entre os dois serviços é definida em runtime pelo comando executado.

Para subir os dois serviços via Docker Compose:

```bash
docker compose up --build
```

Portas utilizadas:

- **Streamlit:** `8501`
- **API:** `8000`

Para parar os serviços:

```bash
docker compose down
```

## API

A API REST (FastAPI) expõe, entre outros recursos, documentação interativa em:

```
http://localhost:8000/docs
```

Endpoints existentes:

- **`GET /health`** — verificação simples de disponibilidade do processo da API, sem depender de configuração externa (ex.: chave da Anthropic).
- **`POST /analises`** — recebe um arquivo CSV e executa o pipeline completo de análise (limpeza, análise estatística, visualizações e relatório), devolvendo o resultado completo.
- **`POST /perguntas`** — recebe uma pergunta em linguagem natural e um arquivo CSV, e responde via `Agent` com tool-calling sobre o dataset enviado. Requer `DAA_ANTHROPIC_API_KEY` configurada; sem ela, a requisição retorna erro de configuração.

## Streamlit

A interface Streamlit permite carregar um CSV, executar a análise, visualizar os resultados (limpeza, estatísticas, gráficos), gerar o relatório e exportá-lo (Markdown, HTML ou PDF), além de interagir com o Chat quando o provider Anthropic está configurado.

Sem `DAA_ANTHROPIC_API_KEY`, o Chat fica indisponível, mas todas as funcionalidades determinísticas (limpeza, análise, visualização, relatório e exportações) continuam disponíveis normalmente.

## Chat com IA

O Chat utiliza o provider Anthropic, configurado através de:

- `DAA_ANTHROPIC_API_KEY`
- `DAA_ANTHROPIC_MODEL`

A integração é feita através do `Agent`, que usa tool-calling para executar as operações disponíveis (ex.: consultar o dataset, disparar etapas do pipeline) em resposta a perguntas em linguagem natural. Nenhuma chave de API real deve constar neste repositório.

## Exportações

Formatos existentes:

- **Markdown**
- **HTML** (autocontido, para uso offline)
- **PDF**

O conteúdo em Markdown é gerado centralmente em `engine/report.py`; os demais exporters (HTML e PDF) reutilizam essa mesma representação como base, evitando duplicação da lógica de geração de relatório.

## Testes

Executar toda a suíte de testes:

```bash
pytest -q
```

Existem testes unitários, de integração e de performance. Os testes de performance ficam isolados por um marker dedicado; para rodar a suíte sem eles:

```bash
pytest -m "not performance"
```

## CI

O projeto possui um workflow de GitHub Actions (`.github/workflows/ci.yml`) que, a cada push ou pull request:

- instala o projeto com as dependências de desenvolvimento (`pip install -e ".[dev]"`);
- executa os testes excluindo os marcados como `performance`;
- valida o build da imagem Docker definida no `Dockerfile`.

A CI não publica a imagem em nenhum registry e não realiza deploy.

## Docker / arquitetura de execução

API e Streamlit utilizam a mesma imagem Docker; o comando de execução de cada serviço é definido no `docker-compose.yml`, sem necessidade de manter Dockerfiles separados.

## Status do projeto

As Fases 1–8 do roadmap aprovado no SDD v2.0 foram concluídas:

1. Streamlit + dashboard + logging/exceções.
2. Interface com LLM/provider.
3. Chat com tool-calling.
4. Arquitetura formal do orquestrador (`Agent`).
5. API REST (FastAPI).
6. Exportação de relatórios (Markdown, HTML, PDF).
7. Docker e Docker Compose.
8. CI (GitHub Actions).

Deployment em produção, publicação de imagem em registry e hospedagem **não** estão implementados neste projeto.

## Roadmap

Com as Fases 1–8 concluídas, este momento representa a etapa de fechamento e documentação do projeto — não a introdução de uma nova fase técnica.

## Observações

- O pipeline determinístico (limpeza, análise, visualização, relatório e exportações) funciona sem a chave da Anthropic.
- O Chat e a geração de insights via IA requerem `DAA_ANTHROPIC_API_KEY` configurada.
- O arquivo `.env` não deve ser versionado.