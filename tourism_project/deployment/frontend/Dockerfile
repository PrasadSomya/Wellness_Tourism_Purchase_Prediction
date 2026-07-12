FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
