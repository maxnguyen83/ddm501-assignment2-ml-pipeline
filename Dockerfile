# Reproducible runtime for the training pipeline.
#   docker build -t pm-pipeline:0.3.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#   docker run --rm -v "$PWD/mlruns-docker:/app/state" pm-pipeline:0.3.0 all
FROM python:3.12-slim AS runtime

# libgomp1: OpenMP runtime needed by LightGBM wheels on Debian slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    GIT_COMMIT=${GIT_COMMIT} \
    PM_CONFIG=/app/configs/pipeline.yaml \
    PM_EXPERIMENTS=/app/configs/experiments.yaml \
    PM_TRACKING__TRACKING_URI=sqlite:////app/state/mlflow.db \
    PM_TRACKING__ARTIFACT_LOCATION=/app/state/mlartifacts \
    PM_DATA__WORK_DIR=/app/state/interim

WORKDIR /app
# Dependencies first (pinned lock file) so code edits do not invalidate this layer.
COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY data/raw ./data/raw

RUN useradd --create-home --uid 10001 pipeline \
    && mkdir -p /app/state /app/reports \
    && chown -R pipeline /app/state /app/reports
USER pipeline

ENTRYPOINT ["pm-pipeline"]
CMD ["--help"]
