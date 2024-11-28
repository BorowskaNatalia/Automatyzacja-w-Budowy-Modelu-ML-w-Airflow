from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Ścieżka do pliku JSON z danymi uwierzytelniającymi
json_path = "airflow-442316-8c5dfa0cf9c0.json"
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


# Funkcja do pobrania danych modelowych z Google Sheets
def fetch_model_data_from_sheets(**kwargs):
    creds = Credentials.from_service_account_file(json_path, scopes=scopes)
    client = gspread.authorize(creds)

    try:
        sheet = client.open("Airflow").worksheet("Zbiór modelowy")
        data = pd.DataFrame(sheet.get_all_records())
        kwargs['ti'].xcom_push(key='data', value=data.to_json())
    except gspread.exceptions.SpreadsheetNotFound:
        logging.error("Arkusz nie został znaleziony. Sprawdź nazwę i dostęp.")
    except gspread.exceptions.APIError as e:
        logging.error(f"Problem z API: {e}")
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd: {e}")


# Funkcja do czyszczenia danych
def clean_data(**kwargs):
    data_json = kwargs['ti'].xcom_pull(key='data')
    data = pd.read_json(data_json)

    # Czyszczenie danych - usunięcie wartości brakujących i duplikatów
    data.dropna(inplace=True)  # Usunięcie brakujących wartości
    data.drop_duplicates(inplace=True)  # Usunięcie duplikatów

    kwargs['ti'].xcom_push(key='clean_data', value=data.to_json())


# Funkcja do standaryzacji i normalizacji danych
def standardize_and_normalize_data(**kwargs):
    # Pobranie danych z XCom
    data_json = kwargs['ti'].xcom_pull(key='clean_data')

    # Sprawdzenie, czy dane zostały pobrane
    if not data_json:
        logging.error("Nie udało się pobrać danych z XCom. Sprawdź poprzednie taski i upewnij się, że dane są prawidłowo przekazywane.")
        return

    try:
        # Konwersja JSON na DataFrame
        data = pd.read_json(data_json)

        # Logowanie danych przed przetwarzaniem
        logging.info("Dane przed standaryzacją i normalizacją:\n%s", data.head())

        # Wybór tylko numerycznych kolumn do przetwarzania
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        # Sprawdzenie, czy istnieją numeryczne dane do przetworzenia
        if numeric_data.empty:
            logging.warning("Brak kolumn numerycznych do standaryzacji i normalizacji.")
            kwargs['ti'].xcom_push(key='processed_data', value=data.to_json())
            return

        # Standaryzacja danych
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)

        # Normalizacja danych
        normalizer = MinMaxScaler()
        data_normalized = normalizer.fit_transform(data_scaled)

        # Utworzenie DataFrame ze znormalizowanymi danymi
        data[numeric_data.columns] = data_normalized

        # Logowanie przetworzonych danych
        logging.info("Dane po standaryzacji i normalizacji:\n%s", data.head())

        # Przekazanie przetworzonych danych do XCom
        kwargs['ti'].xcom_push(key='processed_data', value=data.to_json())
        logging.info("Dane zostały pomyślnie standaryzowane, znormalizowane i przekazane do XCom.")

    except Exception as e:
        logging.error(f"Błąd podczas standaryzacji i normalizacji danych: {e}")


# Funkcja do zapisu przetworzonych danych do Google Sheets
def upload_processed_data_to_google_sheets(**kwargs):
    creds = Credentials.from_service_account_file(json_path, scopes=scopes)
    client = gspread.authorize(creds)

    processed_data = pd.read_json(kwargs['ti'].xcom_pull(key='processed_data'))

    try:
        # Otwieramy istniejący arkusz Google "Airflow"
        spreadsheet = client.open("Airflow")

        # Dodajemy lub wybieramy arkusz dla przetworzonych danych
        if "Dane przetworzone" not in [worksheet.title for worksheet in spreadsheet.worksheets()]:
            processed_worksheet = spreadsheet.add_worksheet(title="Dane przetworzone", rows="100", cols="20")
        else:
            processed_worksheet = spreadsheet.worksheet("Dane przetworzone")

        # Aktualizujemy dane przetworzone
        processed_worksheet.update([processed_data.columns.values.tolist()] + processed_data.values.tolist())

    except gspread.exceptions.APIError as e:
        logging.error(f"Problem z API: {e}")
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd: {e}")


# Definicja DAG-a
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

with DAG(
        'data_processing_dag_v2',
        default_args=default_args,
        description='DAG do przetwarzania danych z Google Sheets',
        schedule_interval=None,
        catchup=False,
) as dag:
    # Task 1: Pobranie danych modelowych z Google Sheets
    fetch_data = PythonOperator(
        task_id='fetch_model_data',
        python_callable=fetch_model_data_from_sheets,
        provide_context=True
    )

    # Task 2: Czyszczenie danych
    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True
    )

    # Task 3: Standaryzacja i normalizacja danych
    standardize_normalize_task = PythonOperator(
        task_id='standardize_and_normalize_data',
        python_callable=standardize_and_normalize_data,
        provide_context=True
    )

    # Task 4: Zapisanie przetworzonych danych do Google Sheets
    upload_data_task = PythonOperator(
        task_id='upload_processed_data',
        python_callable=upload_processed_data_to_google_sheets,
        provide_context=True
    )

    # Ustalanie kolejności tasków w DAG-u
    fetch_data >> clean_data_task >> standardize_normalize_task >> upload_data_task
