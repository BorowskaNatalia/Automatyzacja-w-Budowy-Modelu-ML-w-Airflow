from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pickle
import os


# Funkcja do trenowania modelu
def train_model():
    # Wczytaj przetworzone dane
    df = pd.read_csv('/opt/airflow/processed_data/processed_data_train.csv')

    # Sprawdź, czy kolumna docelowa istnieje
    target_column = 'Have you ever had suicidal thoughts ?'
    if target_column not in df.columns:
        raise ValueError(f"Kolumna docelowa '{target_column}' nie istnieje w danych.")

    # Wydziel kolumny wejściowe (X) oraz kolumnę docelową (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Kodowanie wartości tekstowych na wartości liczbowe
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

    # Kodowanie wartości tekstowych w kolumnie docelowej, jeśli jest typu tekstowego
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Uzupełnianie brakujących wartości za pomocą SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Podziel dane na zbiór treningowy i testowy (70% trening, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Trenuj model logistycznej regresji
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Ewaluacja modelu na zbiorze testowym
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Tworzenie folderu na modele, jeśli nie istnieje
    os.makedirs('/opt/airflow/models', exist_ok=True)

    # Zapisz model do pliku w formacie pickle
    with open('/opt/airflow/models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Tworzenie folderu na raporty, jeśli nie istnieje
    os.makedirs('/opt/airflow/reports', exist_ok=True)

    # Zapisz raport ewaluacji modelu do pliku tekstowego
    with open('/opt/airflow/reports/evaluation_report.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}')


# Definicja DAG dla trenowania modelu ML
with DAG(
        dag_id='model_training_dag',
        start_date=datetime(2024, 10, 1),
        schedule_interval=None,
        catchup=False
) as dag:
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
