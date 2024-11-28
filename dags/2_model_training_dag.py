from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Funkcja do trenowania modelu
def train_model(**kwargs):
    # Wczytanie danych
    data_path = '/opt/airflow/processed_data/processed_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop('Have you ever had suicidal thoughts ?_Yes', axis=1)
    y = data['Have you ever had suicidal thoughts ?_Yes']

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Przygotowanie pipeline'u
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Parametry do GridSearch
    param_grid = [
        {
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1.0, 10.0]
        },
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [10, 50, 100]
        },
        {
            'classifier': [SVC()],
            'classifier__C': [0.1, 1.0, 10.0]
        }
    ]

    # GridSearchCV do znalezienia najlepszego modelu
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Najlepszy model i jego ewaluacja
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Zapisz model
    with open('/opt/airflow/models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Zapisz raport ewaluacji
    with open('/opt/airflow/reports/evaluation_report.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Best parameters: {grid_search.best_params_}\n')

# Definicja DAG-a
with DAG(
    dag_id='2_model_training_dag',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='DAG do trenowania modelu ML',
    schedule_interval=None,
    start_date=datetime(2023, 11, 28),
    catchup=False,
) as dag:

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )
