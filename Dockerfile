# syntax=docker/dockerfile:1
#
# Fase 7 (SDD v2.0/v2.1) — imagem única para os dois adaptadores finos do
# projeto (Streamlit e API FastAPI). Qual dos dois sobe é decidido em
# runtime, via `command`/`docker run <imagem> <comando>` — nenhum dos dois
# é definido implicitamente além do CMD padrão abaixo (API).
#
# Escopo desta etapa: só o suficiente para build + instalação de
# dependências + arranque de cada serviço. Sem docker-compose.yml, sem
# volumes, sem execução de pytest no build (decisões já fechadas).

FROM python:3.14-slim

WORKDIR /app

# Copia apenas o que o build (hatchling) precisa para instalar o pacote:
# pyproject.toml declara as dependências e aponta README.md como
# `readme`; src/ é o próprio pacote a ser empacotado.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Instalação não-editável e só de dependências de produção — sem o extra
# [dev] (pytest/pytest-cov/httpx não entram na imagem final).
RUN pip install --no-cache-dir .

# Portas dos dois serviços possíveis: Streamlit (8501) e API/uvicorn (8000).
EXPOSE 8501 8000

# Padrão: sobe a API. Para o Streamlit, sobrescrever o comando, ex.:
#   docker run -p 8501:8501 <imagem> \
#     python -m streamlit run src/data_analysis_agent/interfaces/streamlit_app/app.py \
#     --server.address=0.0.0.0 --server.port=8501 --server.headless=true
CMD ["python", "-m", "uvicorn", "data_analysis_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]