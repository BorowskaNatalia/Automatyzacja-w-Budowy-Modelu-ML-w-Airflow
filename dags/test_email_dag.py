from airflow import DAG
from airflow.operators.email import EmailOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 12, 4),
    'email_on_failure': True,
    'email_on_retry': False,
}

with DAG(
    dag_id='test_email_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    send_email = EmailOperator(
        task_id='send_email',
        to='test@example.com',
        subject='Test email from Airflow',
        html_content='<h3>Airflow Email Test</h3><p>This is a test email sent from Airflow DAG.</p>',
    )

    send_email
