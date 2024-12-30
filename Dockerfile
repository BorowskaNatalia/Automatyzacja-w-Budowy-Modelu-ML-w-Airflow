# Bazujemy na Pythonie 3.8, aby uniknąć problemów z wersjami
FROM python:3.8-slim-buster

# Instalujemy potrzebne zależności systemowe oraz PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ustawiamy zmienną środowiskową AIRFLOW_HOME
ENV AIRFLOW_HOME=/opt/airflow

# Instalujemy Airflow
RUN pip install apache-airflow==2.5.1



# Kopiujemy pliki DAG, logi, pluginy oraz inne wymagane pliki
COPY ./dags $AIRFLOW_HOME/dags
COPY ./logs $AIRFLOW_HOME/logs
COPY ./plugins $AIRFLOW_HOME/plugins
COPY ./requirements.txt /requirements.txt
COPY app/rest-api-service.py /rest-api-service.py
COPY ./airflow.cfg $AIRFLOW_HOME/airflow.cfg

RUN pip install --upgrade pip


# Instalujemy wymagane zależności z requirements.txt
RUN pip install -r /requirements.txt
RUN pip install --upgrade flask-session
# Inicjujemy Airflow
RUN airflow db init && \
    airflow users create -r Admin -u admin -p admin -e admin@example.com -f Admin -l User

# Ustawiamy polecenie uruchamiające
CMD ["airflow", "webserver"]
