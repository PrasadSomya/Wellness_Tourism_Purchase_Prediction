FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .

RUN useradd --create-home --uid 1000 user \
    && chown -R user:user /app

USER user
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
