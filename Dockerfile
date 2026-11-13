FROM apache/airflow:2.9.2-python3.12

# xgboost a besoin de la bibliothèque système libgomp
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt

# Le fichier "constraint" garde les versions compatibles avec Airflow 2.9.2
RUN pip install --no-cache-dir \
      "apache-airflow==2.9.2" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.12.txt" \
      -r /requirements.txt

# Pour que "python -m src.train" trouve le package src
ENV PYTHONPATH=/opt/airflow