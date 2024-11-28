from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_data():
    file_path = 'processed_data/data.csv'
    data = pd.read_csv(file_path)

    # Podstawowa analiza danych (EDA)
    data.info()
    data.describe()

    # Sprawdzenie brakujących wartości
    missing_values = data.isnull().sum()

    # Zamiana wartości tekstowych na NaN dla kolumn numerycznych
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Usunięcie kolumn z dużą ilością brakujących danych (ponad 50%)
    threshold = 0.5 * len(data)
    columns_to_drop = missing_values[missing_values > threshold].index.tolist()
    columns_to_drop.append('Name')  # Usunięcie kolumny 'Name'
    columns_to_drop.append('City')  # Usunięcie kolumny 'Name'
    data_cleaned = data.drop(columns=columns_to_drop)

    # Uzupełnianie brakujących wartości medianą
    numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    data_cleaned[numerical_columns] = data_cleaned[numerical_columns].fillna(data_cleaned[numerical_columns].median())

    # Kodowanie kolumn kategorycznych
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

    # Zapis przetworzonych danych do folderu processed_data
    data_encoded.to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)

# Definicja DAG
with DAG(
    dag_id='1_data_processing_dag',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='DAG do przetwarzania danych',
    schedule_interval=None,
    start_date=datetime(2023, 11, 26),
    catchup=False,
) as dag:

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )
