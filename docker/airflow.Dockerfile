# Airflow 2.8 image with the pipeline package installed, used to run the DAG end to end.
# numpy/pandas/SQLAlchemy are pinned to the versions shipped with the Airflow image so that
# installing the ML stack cannot upgrade a library Airflow itself depends on.
FROM apache/airflow:2.8.4-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY --chown=airflow:root pyproject.toml README.md /opt/airflow/project/
COPY --chown=airflow:root src /opt/airflow/project/src
RUN pip install --no-cache-dir \
        "apache-airflow==2.8.4" \
        "numpy==1.24.4" "pandas==2.1.4" "sqlalchemy==1.4.52" \
        "/opt/airflow/project"

ENV PM_PROJECT_ROOT=/opt/airflow/project \
    PM_CONFIG=/opt/airflow/project/configs/pipeline.yaml \
    PM_EXPERIMENTS=/opt/airflow/project/configs/experiments.yaml
