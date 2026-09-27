"""Airflow DAG for the aus_yolo_mlops pipeline.

Every task is one `run_cli.py <stage> <run_id>` invocation, exactly the command
you would run by hand, so the DAG and the CLI can never drift apart. The run id
is a DAG param: trigger the DAG with {"run_id": "run_002"} to keep runs apart.

  start -> cleanup_previous_run -> fetch_dataset -> fetch_region
        -> prepare_data -> build_dataset
        -> train_segmentation -> export_onnx -> quantize_model
        -> finalize_results -> compare_groundtruth -> predict_area
        -> log_mlflow -> upload_outputs -> end
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

PROJECT_ROOT = os.environ.get("MLOPS_PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get("MLOPS_PYTHON", "python")

STAGES = [
    "cleanup_previous_run",
    "fetch_dataset",
    "fetch_region",
    "prepare_data",
    "build_dataset",
    "train_segmentation",
    "export_onnx",
    "quantize_model",
    "finalize_results",
    "compare_groundtruth",
    "predict_area",
    "log_mlflow",
    "upload_outputs",
]

default_args = {
    "owner": "aus-yolo-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="aus_yolo_mlops_pipeline",
    description=("Australia building footprints -> leakage-safe YOLOv8-seg "
                 "dataset -> train -> ONNX/INT8 -> evaluate -> track/upload"),
    schedule=None,
    catchup=False,
    start_date=datetime(2026, 1, 1),
    params={"run_id": "run_001"},
    default_args=default_args,
    tags=["yolo", "mlops", "australia"],
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    previous = start
    for stage in STAGES:
        task = BashOperator(
            task_id=stage,
            bash_command=f'"{PYTHON}" run_cli.py {stage} "$RUN_ID"',
            env={"RUN_ID": "{{ params.run_id }}"},
            cwd=PROJECT_ROOT,
        )
        previous >> task
        previous = task
    previous >> end
