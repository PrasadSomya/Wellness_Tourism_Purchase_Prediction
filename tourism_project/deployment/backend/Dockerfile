FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
