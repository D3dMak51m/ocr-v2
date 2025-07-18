FROM apache/airflow:2.7.3

ENV AIRFLOW_HOME=/opt/airflow


USER root


COPY dags/ /opt/airflow/dags/
COPY requirements.txt /requirements.txt


USER airflow
RUN pip install -r /requirements.txt


