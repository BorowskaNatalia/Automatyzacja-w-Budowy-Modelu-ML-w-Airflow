from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import logging

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ścieżka do modelu i danych
MODEL_FILE_PATH = '/opt/airflow/models/model.pkl'
NEW_DATA_FILE_PATH = '/opt/airflow/processed_data/processed_data.csv'
CRITICAL_ACCURACY_THRESHOLD = 0.80


# Funkcja do ładowania modelu
def load_model():
    if os.path.exists(MODEL_FILE_PATH):
        with open(MODEL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model został poprawnie załadowany.")
        return model
    else:
        raise FileNotFoundError(f'Nie znaleziono pliku modelu w ścieżce: {MODEL_FILE_PATH}')


# Funkcja do oceny jakości modelu na nowych danych
def evaluate_model(**kwargs):
    # Ładowanie danych
    try:
        new_data = pd.read_csv(NEW_DATA_FILE_PATH)
        if new_data.empty:
            raise ValueError("Nowe dane są puste. Walidacja nie może zostać wykonana.")

        # Dopasowanie nazw kolumn do modelu - usunięcie niepotrzebnych kolumn
        columns_to_drop = ['Name', 'City']
        columns_to_drop = [col for col in columns_to_drop if col in new_data.columns]
        if 'Have you ever had suicidal thoughts ?_Yes' in new_data.columns:
            target_column = 'Have you ever had suicidal thoughts ?_Yes'
        elif 'target_column' in new_data.columns:
            target_column = 'target_column'
        else:
            raise KeyError("Kolumna docelowa nie została znaleziona w danych.")

        X_new = new_data.drop(columns=columns_to_drop + [target_column])
        y_new = new_data[target_column]

        # Ładowanie modelu
        model = load_model()

        # Dopasowanie liczby funkcji do modelu
        if X_new.shape[1] != model.n_features_in_:
            X_new = X_new.iloc[:, :model.n_features_in_]

        # Przewidywanie i ocena
        y_pred = model.predict(X_new)
        accuracy = accuracy_score(y_new, y_pred)
        precision = precision_score(y_new, y_pred, zero_division=1)
        recall = recall_score(y_new, y_pred, zero_division=1)

        # Zapis wyników do XCom
        kwargs['ti'].xcom_push(key='model_accuracy', value=accuracy)
        kwargs['ti'].xcom_push(key='model_precision', value=precision)
        kwargs['ti'].xcom_push(key='model_recall', value=recall)
        logger.info(
            f"Model został oceniony z dokładnością: {accuracy:.2f}, precyzją: {precision:.2f}, recall: {recall:.2f}")

        # Sprawdzenie, czy jakość modelu jest poniżej krytycznego progu i zapisanie statusu do XCom
        if accuracy < CRITICAL_ACCURACY_THRESHOLD:
            kwargs['ti'].xcom_push(key='model_status', value='failed')
            logger.warning(
                f"Jakość modelu (accuracy) spadła poniżej progu krytycznego: {accuracy:.2f} < {CRITICAL_ACCURACY_THRESHOLD}")
        else:
            kwargs['ti'].xcom_push(key='model_status', value='passed')

    except FileNotFoundError:
        raise FileNotFoundError(f"Nie znaleziono nowego pliku danych w ścieżce: {NEW_DATA_FILE_PATH}")
    except Exception as e:
        kwargs['ti'].xcom_push(key='model_status', value='failed')
        logger.error(f"Wystąpił błąd podczas oceny: {e}")


# Funkcja do testów modelu
def run_tests(**kwargs):
    failed_tests = []

    # Wczytanie modelu
    model = load_model()

    # Test: Czy model ładuje dane i przewiduje wyniki
    try:
        test_data = pd.read_csv(NEW_DATA_FILE_PATH).drop(
            columns=['Name', 'City', 'Have you ever had suicidal thoughts ?_Yes'], errors='ignore')
        model.predict(test_data.iloc[:, :model.n_features_in_])
        logger.info("Model poprawnie przewiduje wyniki na danych testowych.")
    except Exception as e:
        failed_tests.append(f"Model nie powiódł się podczas predykcji: {e}")

    # Test: Czy pipeline walidacji obsługuje przypadki braku danych
    try:
        empty_data = pd.DataFrame()
        model.predict(empty_data)
    except ValueError:
        # Oczekiwane zachowanie: model powinien rzucić ValueError
        logger.info("Model poprawnie obsługuje przypadek braku danych.")
    except Exception as e:
        failed_tests.append(f"Nieoczekiwany błąd podczas testowania obsługi pustych danych: {e}")

    # Zapisanie informacji o nieudanych testach do XCom
    if failed_tests:
        kwargs['ti'].xcom_push(key='failed_tests', value=failed_tests)
        kwargs['ti'].xcom_push(key='test_status', value='failed')
        logger.error("Niektóre testy nie przeszły pomyślnie.")
    else:
        kwargs['ti'].xcom_push(key='test_status', value='passed')


# Definicja DAG
with DAG(
        dag_id='model_validation_monitoring_dag',
        default_args={
            'owner': 'airflow',
            'depends_on_past': False,
            'email_on_failure': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        },
        description='DAG do walidacji i monitorowania modelu ML z powiadomieniem mailowym',
        schedule_interval=None,
        start_date=datetime(2023, 11, 28),
        catchup=False,
) as dag:
    # Task 1: Ocena jakości modelu na nowych danych
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )

    # Task 2: Uruchomienie testów modelu
    run_tests_task = PythonOperator(
        task_id='run_model_tests',
        python_callable=run_tests,
        provide_context=True,
    )

    # Task 3: Wysłanie powiadomienia mailowego, jeśli jakość modelu spadnie poniżej progu lub testy nie przejdą
    send_email_task = EmailOperator(
        task_id='send_email_notification',
        to='n.borowska@huma.waw.pl',
        subject='[Alert] Spadek jakości modelu ML poniżej krytycznego progu',
        html_content="""
        <h3>Uwaga!</h3>
        <p>Model ML nie spełnia kryterium jakości lub testy modelu zakończyły się niepowodzeniem.</p>
        <ul>
            <li>Nazwa modelu: model.pkl</li>
            <li>Aktualna jakość (Accuracy): {{ task_instance.xcom_pull(task_ids='evaluate_model', key='model_accuracy') }}</li>
            <li>Aktualna precyzja (Precision): {{ task_instance.xcom_pull(task_ids='evaluate_model', key='model_precision') }}</li>
            <li>Aktualny recall: {{ task_instance.xcom_pull(task_ids='evaluate_model', key='model_recall') }}</li>
            <li>Krytyczny próg: 80%</li>
            <li>Nieudane testy: {{ task_instance.xcom_pull(task_ids='run_model_tests', key='failed_tests') }}</li>
        </ul>
        <p>Sprawdź szczegóły i podejmij odpowiednie działania.</p>
        """,
        trigger_rule='one_failed',  # Wysyłanie emaila, jeśli poprzedni task zakończył się błędem
    )

    # Definicja zależności między taskami
    evaluate_model_task >> run_tests_task >> send_email_task
