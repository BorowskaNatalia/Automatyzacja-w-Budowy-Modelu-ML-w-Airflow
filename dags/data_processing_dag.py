from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ścieżka do pliku JSON z danymi uwierzytelniającymi dla Google API
SERVICE_ACCOUNT_FILE = 'airflow-442316-8c5dfa0cf9c0.json'

# Zakresy uprawnień do odczytu/zapisu w arkuszach Google
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
PROCESSED_DATA_PATH = '/opt/airflow/processed_data/'

def download_data(**kwargs):
    # Funkcja pobiera dane z pliku CSV
    file_path = kwargs.get('file_path', 'data.csv')
    try:
        data = pd.read_csv(file_path)
        data_copy = data.copy()
        # Usuwanie niepotrzebnych kolumn 'Name' i 'City'
        data = data.drop(columns=['Name', 'City'])
        kwargs['ti'].xcom_push(key='data', value=data.to_json())
    except FileNotFoundError:
        print("Plik nie został znaleziony. Sprawdź podaną ścieżkę.")
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")


def split_data(**kwargs):
    # Funkcja dzieli dane na zbiór modelowy i douczeniowy
    data_json = kwargs['ti'].xcom_pull(key='data')
    data = pd.read_json(data_json)
    data_copy = data.copy()

    # Standaryzacja danych - usuwamy kolumnę celu "Depression", jeśli istnieje
    if 'Have you ever had suicidal thoughts ?' in data_copy.columns:
        X = data_copy.drop('Have you ever had suicidal thoughts ?', axis=1)
        y = data_copy['Have you ever had suicidal thoughts ?']
    else:
        X = data_copy
        y = None

    # Skalowanie cech (tylko numerycznych kolumn)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

    # Tworzenie DataFrame z wyskalowanymi danymi i dodanie kolumn nie-numerowych, jeśli istnieją
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=['float64', 'int64']).columns)
    for column in X.select_dtypes(exclude=['float64', 'int64']).columns:
        X_scaled_df[column] = X[column].values

    # Dodanie kolumny celu, jeśli istnieje
    if y is not None:
        X_scaled_df['Have you ever had suicidal thoughts ?'] = y

    # Podział na 70% - modelowy, 30% - douczeniowy
    train, test = train_test_split(X_scaled_df, test_size=0.3, random_state=42)

    # Przekazanie wyników do XCom
    kwargs['ti'].xcom_push(key='train_data', value=train.to_json())
    kwargs['ti'].xcom_push(key='test_data', value=test.to_json())

    if not data_json:
        raise ValueError("Nie udało się pobrać danych z XCom - brak danych.")

    try:
        data = pd.read_json(data_json)
        if data.empty:
            raise ValueError("Dane do podziału są puste.")

        # Podział na 70% - modelowy, 30% - douczeniowy
        train, test = train_test_split(data, test_size=0.3, random_state=42)

        # Zapisanie zbiorów do plików CSV
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        train.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_data_train.csv'), index=False)
        test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_data_test.csv'), index=False)

        # Przekazanie wyników do XCom
        kwargs['ti'].xcom_push(key='train_data', value=train.to_json())
        kwargs['ti'].xcom_push(key='test_data', value=test.to_json())
        print("Dane zostały podzielone i zapisane do XCom.")
    except Exception as e:
        print(f"Błąd podczas podziału danych: {e}")


def upload_to_google_sheets(**kwargs):
    # Autoryzacja do Google Sheets
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)

    # Pobieranie danych z XCom
    train_data_json = kwargs['ti'].xcom_pull(key='train_data')
    test_data_json = kwargs['ti'].xcom_pull(key='test_data')

    # Konwersja JSON na DataFrame
    train = pd.read_json(train_data_json)
    test = pd.read_json(test_data_json)

    # Przed zapisaniem danych, zastępujemy wartości NaN i +/-inf na 0
    train = train.replace([pd.NA, float('inf'), float('-inf')], 0)
    test = test.replace([pd.NA, float('inf'), float('-inf')], 0)

    # Otwieranie lub tworzenie arkusza Google
    try:
        spreadsheet = client.open("Airflow")
    except gspread.SpreadsheetNotFound:
        spreadsheet = client.create("Airflow")

    # Zapis zbioru modelowego do pierwszej karty
    try:
        train_sheet = spreadsheet.worksheet("Zbiór modelowy")
    except gspread.WorksheetNotFound:
        train_sheet = spreadsheet.add_worksheet(title="Zbiór modelowy", rows=str(len(train) + 1),
                                                cols=str(train.shape[1]))

    train_sheet.clear()  # Czyszczenie istniejącej karty przed zapisem nowych danych
    train_sheet.update([train.columns.values.tolist()] + train.fillna('').values.tolist())

    # Zapis zbioru douczeniowego do drugiej karty
    try:
        test_sheet = spreadsheet.worksheet("Zbiór douczeniowy")
    except gspread.WorksheetNotFound:
        test_sheet = spreadsheet.add_worksheet(title="Zbiór douczeniowy", rows=str(len(test) + 1),
                                               cols=str(test.shape[1]))

    test_sheet.clear()  # Czyszczenie istniejącej karty przed zapisem nowych danych
    test_sheet.update([test.columns.values.tolist()] + test.fillna('').values.tolist())


# Definicja DAG
with DAG(
        dag_id='data_split_google_sheets',
        default_args={
            'owner': 'airflow',
            'depends_on_past': False,
            'email_on_failure': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        },
        description='DAG do pobrania, podziału i zapisu danych do Google Sheets',
        schedule_interval=None,
        start_date=datetime(2023, 11, 26),
        catchup=False,
) as dag:
    # Task 1: Pobranie danych z pliku CSV
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        provide_context=True,
        op_kwargs={'file_path': 'data.csv'},
    )

    # Task 2: Podział danych na zbiór modelowy i douczeniowy
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True,
    )

    # Task 3: Zapis podzielonych danych do Google Sheets
    upload_data_task = PythonOperator(
        task_id='upload_to_google_sheets',
        python_callable=upload_to_google_sheets,
        provide_context=True,
    )

    # Definicja zależności między taskami
    download_data_task >> split_data_task >> upload_data_task
