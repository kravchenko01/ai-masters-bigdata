from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.decorators import task
from airflow.sensors.filesystem import FileSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

pyspark_python = "/opt/conda/envs/dsenv/bin/python"

with DAG(
        dag_id="kravchenko01_dag",
        start_date=datetime(2024, 5, 4),
        schedule_interval=None,
        catchup=False,
        description="Это DAG для 6 домашки",
        doc_md = """
        Это учебный DAG. Не надо его переносить в продакшен!
        """,
        tags=["example"],
) as dag:
    base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

    feature_eng_train_task = SparkSubmitOperator(
        task_id="feature_eng_train_task",
        application=f"{base_dir}/dataset_preprocessing.py",
        application_args=["--path-in", '/datasets/amazon/amazon_extrasmall_train.json', "--path-out",  'kravchenko01_train_out'],
        spark_binary="/usr/bin/spark3-submit",
        env_vars={"PYSPARK_PYTHON": pyspark_python},
    )

    download_train_task = BashOperator(
        task_id='download_train_task',
        bash_command=f'hdfs dfs -getmerge kravchenko01_train_out {base_dir}/kravchenko01_train_out_local'
    )

    train_task = BashOperator(
        task_id='train_task',
        bash_command=f'{pyspark_python} {base_dir}/train_model.py --train-in {base_dir}/kravchenko01_train_out_local --sklearn-model-out {base_dir}/6.joblib'
    )

    model_sensor = FileSensor(
        task_id=f'model_sensor',
        filepath=f"{base_dir}/6.joblib",
        poke_interval=30,
        timeout=60 * 5,
    )

    feature_eng_test_task = SparkSubmitOperator(
        task_id="feature_eng_test_task",
        application=f"{base_dir}/dataset_preprocessing.py",
        application_args=["--path-in", '/datasets/amazon/amazon_extrasmall_test.json', "--path-out",  'kravchenko01_test_out'],
        spark_binary="/usr/bin/spark3-submit",
        env_vars={"PYSPARK_PYTHON": pyspark_python},
    )

    predict_task = SparkSubmitOperator(
        task_id="predict_task",
        application=f"{base_dir}/infer_model.py",
        application_args=["--test-in", f'kravchenko01_test_out', "--pred-out", f'kravchenko01_hw6_prediction', '--sklearn-model-in', f'{base_dir}/6.joblib'],
        spark_binary="/usr/bin/spark3-submit",
        env_vars={"PYSPARK_PYTHON": pyspark_python},
    )


    feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task    
