# What Else Was On: the replay page and the agent team in one container.
# Build:  docker build -t whatelse .
# Run:    docker run -p 8000:8000 --env-file .env whatelse      (then open http://localhost:8000)
# Azure:  see docs/DEPLOY.md (Azure Container Apps)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    WHATELSE_DATA=/app/data \
    WHATELSE_LIVE_RUNS=20 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY web ./web
# Whatever matches, stories and xT grid you have locally are copied in (data/ is not in git);
# anything missing is made here: matches 1-3 are generated (same seeds, same matches) and the
# xT grid is downloaded from Karun Singh's site.
COPY data ./data
RUN python scripts/prepare_data.py

# run as a normal user, who may write new stories to data/
RUN useradd --create-home app && chown -R app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz')"
CMD ["sh", "-c", "uvicorn whatelse.app:app --host 0.0.0.0 --port ${PORT}"]
