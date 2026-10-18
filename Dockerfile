FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates libseccomp2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
COPY requirements-core.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-core.lock.txt
COPY pyproject.toml ./
COPY src ./src
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then python -m pip install --no-cache-dir '.[ml]'; \
    else python -m pip install --no-cache-dir --no-deps .; fi
COPY config ./config
RUN mkdir -p /app/runtime /models && chown -R appuser:appuser /app/runtime /models
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "cashwatch.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

