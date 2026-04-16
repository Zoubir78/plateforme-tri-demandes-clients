# api/Dockerfile
FROM python:3.10-slim

ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]