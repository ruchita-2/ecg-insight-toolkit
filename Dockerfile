FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/ecg-venv \
    PATH=/opt/ecg-venv/bin:$PATH

WORKDIR /app
RUN pip install --no-cache-dir uv==0.10.3
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-cache

EXPOSE 8000
CMD ["uvicorn", "ecg_workbench.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
