FROM python:3.11.13-slim-bookworm

ENV HOME=/home/rag-audit \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY app ./app
COPY scripts/provision_access_context.py ./scripts/
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps . \
    && groupadd --system --gid 10001 rag-audit \
    && useradd \
        --system \
        --uid 10001 \
        --gid rag-audit \
        --create-home \
        --home-dir /home/rag-audit \
        rag-audit \
    && mkdir -p /app/data/api /app/data/private/evidence \
    && chown -R rag-audit:rag-audit /app /home/rag-audit

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
