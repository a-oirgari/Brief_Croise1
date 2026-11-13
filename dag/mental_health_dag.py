import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # /opt/airflow dans Docker
log = logging.getLogger(__name__)


def notify_failure(context):
    ti = context["task_instance"]
    log.error("ÉCHEC de la tâche '%s' dans le DAG '%s'", ti.task_id, ti.dag_id)


def run(module):
    return f'cd "{PROJECT_ROOT}" && python -m src.{module}'


with DAG(
    dag_id="student_mental_health_pipeline",
    description="Préparation, préprocessing, entraînement et évaluation",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={
        "owner": "student",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": notify_failure,
    },
    tags=["ml", "pedagogique"],
) as dag:

    prepare = BashOperator(task_id="prepare_data", bash_command=run("data_prep"))
    preprocess = BashOperator(task_id="preprocess", bash_command=run("preprocess"))
    train = BashOperator(task_id="train_model", bash_command=run("train"))
    evaluate = BashOperator(task_id="evaluate_model", bash_command=run("evaluate"))

    prepare >> preprocess >> train >> evaluate