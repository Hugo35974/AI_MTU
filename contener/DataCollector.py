import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import requests
from joblib import load
from loguru import logger
from psycopg2 import sql
from sqlalchemy import create_engine, text

SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))

from src.Pretreatment.ConfigLoader import ConfigLoader
from src.Runner.Runner import Run

Main_path = Path(__file__).parents[0]
Project_path = Path(__file__).parents[1]
Timeseries_path = os.path.join(Project_path, "data", "timeseries")

class DataCollector:
    """
    A class for collecting, processing, and predicting electricity price data.

    This class handles database connections, data fetching from an API,
    data insertion into a database, and running predictions using machine learning models.
    """
    def __init__(self):
        self.config = ConfigLoader()
        self.runner = Run()
        self.connection = None
        self.api_url = self.config.api["api_url"]
        self.api_url_latest = self.config.api["api_url_latest"]
        self._init_connection()
        logger.add(sink=self._log_to_db)
        self._create_tables()
        self.inject_historical_data()
        # self._retrain_model(self.config.model_infos["model_1"])
        # self._retrain_model(self.config.model_infos["model_2"])
        # self._retrain_model(self.config.model_infos["model_3"])
        # self._retrain_model(self.config.model_infos["model_4"])
        # self._retrain_model(self.config.model_infos["model_5"])
        # self._retrain_model(self.config.model_infos["model_6"])
        self._predict_historical_data(self.config.model_infos["model_1"])
        self._predict_historical_data(self.config.model_infos["model_2"])
        self._predict_historical_data(self.config.model_infos["model_3"])
        self._predict_historical_data(self.config.model_infos["model_4"])
        self._predict_historical_data(self.config.model_infos["model_5"])
        self._predict_historical_data(self.config.model_infos["model_6"])

    def _init_connection(self):
        """
        Initialize the database connection.

        Establishes a connection to the PostgreSQL database using the configuration
        provided in self.config.bdd. Also sets up an SQLAlchemy engine for ORM operations.
        """
        if not self.connection or self.connection.closed:
            try:
                self.connection = psycopg2.connect(
                    dbname=self.config.bdd["dbname"],
                    user=self.config.bdd["user"],
                    password=self.config.bdd["password"],
                    host=self.config.bdd["host"],
                    port=self.config.bdd["port"]
                )
                self.engine = create_engine(f"postgresql://{self.config.bdd['user']}:{self.config.bdd['password']}@{self.config.bdd['host']}:{self.config.bdd['port']}/{self.config.bdd['dbname']}")
                logger.info("Database connection established successfully.")
            except psycopg2.DatabaseError as e:
                logger.error(f"Database connection error: {e}")
                sys.exit(1)

    def _close_connection(self):
        """
        Close the database connection if it's open.
        """
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed.")

    def _create_tables(self):
        """
        Create necessary database tables if they don't exist.

        This method creates the following tables:
        - models_performance_crossval
        - models_performance
        - sensor_data
        - logs
        """
        queries = [
            self._create_table_query("models_performance_crossval"),
            self._create_table_query("models_performance"),
            self._create_sensor_data_table_query(),
            self._create_logs_table_query(),
        ]
        with self.connection.cursor() as cur:
            for query in queries:
                cur.execute(query)
            self.connection.commit()
        logger.info("Tables created or verified successfully.")

    def _create_table_query(self, name):
        """
        Generate SQL query for creating a table.

        Args:
            name (str): The name of the table to create.

        Returns:
            str: SQL query for creating the specified table.
        """
        additional_columns = self._generate_additional_columns()
        return f"""
        CREATE TABLE IF NOT EXISTS {name} (
            Model VARCHAR(50),
            Horizon FLOAT,
            Mean_CV_Train_Score FLOAT,
            Mean_CV_Test_Score FLOAT,
            R2_Mean FLOAT,
            MAE_Mean FLOAT,
            SRC_Mean FLOAT,
            MSE_Mean FLOAT,
            RMSE_Mean FLOAT,
            MAPE_Mean FLOAT,
            Start_date_trainning TIMESTAMP,
            End_date_trainning TIMESTAMP,
            Lookback FLOAT,
            Duration_seconds FLOAT,
            {', '.join(additional_columns)},
            PRIMARY KEY (Model, Horizon)
        );
        """

    def _create_sensor_data_table_query(self):
        """
        Generate SQL query for creating the sensor_data table.

        Returns:
            str: SQL query for creating the sensor_data table.
        """
        columns = [f"{model['column_name']} FLOAT" for model in self.config.model_infos.values()]
        return f"""
        CREATE TABLE IF NOT EXISTS sensor_data (
            time TIMESTAMP PRIMARY KEY,
            elec_prices_metric FLOAT,
            {', '.join(columns)}
        );
        """

    def _create_logs_table_query(self):
        """
        Generate SQL query for creating the logs table.

        Returns:
            str: SQL query for creating the logs table.
        """
        return """
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP,
            level VARCHAR(10),
            message TEXT
        );
        """

    def _generate_additional_columns(self):
        """
        Generate additional column definitions for models_performance table.

        Returns:
            list: List of additional column definitions.
        """
        metrics = ['MAE', 'R2', 'SRC', 'MSE', 'RMSE', 'MAPE']
        return [f"{metric}_T_{i} FLOAT" for metric in metrics for i in range(24)]

    def _log_to_db(self, record):
        """
        Log a record to the database.

        Args:
            record: The log record to be inserted into the database.
        """
        log_message = record.record['message']
        log_level = record.record["level"].name
        log_time = record.record["time"].timestamp()

        with self.connection.cursor() as cur:
            cur.execute(
                "INSERT INTO logs (time, level, message) VALUES (to_timestamp(%s), %s, %s)",
                (log_time, log_level, log_message)
            )
        self.connection.commit()

    def _fetch_data(self, start_ts, end_ts):
        """
        Fetch data from the API for a given time range.

        Args:
            start_ts (int): Start timestamp in milliseconds.
            end_ts (int): End timestamp in milliseconds.

        Returns:
            dict or None: JSON response from the API if successful, None otherwise.
        """
        params = {'startTs': start_ts, 'endTs': end_ts}
        response = requests.get(self.api_url, params=params)
        if response.status_code == 200:
            logger.info(f"Data fetched successfully from API for period: {start_ts} to {end_ts}.")
            return response.json()
        logger.error(f"Failed to fetch data from API. Status code: {response.status_code}")
        return None

    def _upsert_data(self, df, dynamic_column_name):
        """
        Upsert (insert or update) data into the sensor_data table.

        Args:
            df (pandas.DataFrame): DataFrame containing the data to upsert.
            dynamic_column_name (str): Name of the column to upsert.
        """
        with self.connection.cursor() as cur:
            for idx, row in df.iterrows():
                cur.execute(f"""
                    INSERT INTO sensor_data (time,{dynamic_column_name})
                    VALUES (%s, %s)
                    ON CONFLICT (time)
                    DO UPDATE SET {dynamic_column_name} = COALESCE(EXCLUDED.{dynamic_column_name}, sensor_data.{dynamic_column_name});
                """, (row['time'], row[dynamic_column_name]))
        self.connection.commit()

    def date_to_timestamp(self, date_str):
        """
        Convert a date string to a timestamp in milliseconds.

        Args:
            date_str (str): Date string in format "YYYY/MM/DD".

        Returns:
            int: Timestamp in milliseconds.
        """
        return int(datetime.strptime(date_str, "%Y/%m/%d").timestamp()) * 1000

    def inject_historical_data(self):
        """
        Inject historical data from both API and CSV file into the database.
        """
        try:
            latest_data = self._fetch_latest_data_API()
            start_ts = self.date_to_timestamp("2014/01/01")
            end_ts = int(latest_data.timestamp() * 1000) + 1

            data = self._fetch_data(start_ts, end_ts)
            if data:
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df = df.rename(columns={"value": "elec_prices_metric"})
                self._upsert_data(df,"elec_prices_metric")
                logger.info("API historical data injected successfully.")

            csv_path = os.path.join(Timeseries_path, "Electricity_Prices_Ireland.csv")
            df = pd.read_csv(csv_path)
            df = df.rename(columns={"elec_prices": "elec_prices_metric", "applicable_date": "time"})
            df['time'] = pd.to_datetime(df['time'])
            df = df.dropna()
            self._upsert_data(df,"elec_prices_metric")
            logger.info("CSV data injected successfully.")

        except Exception as e:
            logger.error(f"An error occurred while injecting historical data: {e}")

    def _get_last_100_hours(self, up_to_time=None):
        """
        Retrieve the last 100 hours of data from the sensor_data table.

        Args:
            up_to_time (datetime, optional): The upper time limit for the data retrieval.

        Returns:
            tuple: A numpy array of the last 100 hours of data and the most recent timestamp.
        """
        query = sql.SQL("""
        SELECT time, elec_prices_metric
        FROM sensor_data
        WHERE elec_prices_metric IS NOT NULL 
        {time_condition}
        ORDER BY time DESC
        LIMIT 99;
        """).format(time_condition=sql.SQL("AND time <= %s") if up_to_time else sql.SQL(""))

        with self.connection.cursor() as cur:
            cur.execute(query, (up_to_time,) if up_to_time else None)
            result = cur.fetchall()
            if len(result) < 99:
                logger.warning("Not enough non-null data to predict.")
                return None, None
            times, values = zip(*result)
            return np.array(values), times[0]

    def _predict_next_24_hours(self, last_100_hours, start_time, model_infos):
        """
        Predict the next 24 hours of electricity prices using the specified model.

        Args:
            last_100_hours (numpy.array): Array of the last 100 hours of data.
            start_time (datetime): The start time for the prediction.
            model_infos (dict): Information about the model to use for prediction.

        Returns:
            pandas.DataFrame: DataFrame containing the predictions for the next 24 hours.
        """
        model = load(os.path.join(Main_path, model_infos["path"]))
        date_features = np.array([
            start_time.weekday(), 
            start_time.timetuple().tm_yday,
            start_time.isocalendar()[1],
            start_time.month
        ])
        x = np.concatenate((last_100_hours.ravel(), date_features))
        predictions = model.predict(x.reshape(1, -1))

        future_dates = [start_time + timedelta(hours=i) for i in range(model_infos["horizon"][0], model_infos["horizon"][1]+1)]
        return pd.DataFrame({
            'time': future_dates,
            model_infos["column_name"]: predictions.flatten()
        })

    def _fetch_latest_data_API(self):
        """
        Fetch the latest data from the API.

        Returns:
            datetime or None: The timestamp of the latest data if successful, None otherwise.
        """
        response = requests.get(self.api_url_latest)
        if response.status_code == 200:
            latest_data = response.json()
            logger.info("Latest data fetched from API successfully.")
            return pd.to_datetime(latest_data["price"][0]["ts"], unit='ms', utc=True)
        logger.error(f"Failed to fetch latest data from API. Status code: {response.status_code}")
        return None

    def _fetch_latest_data(self):
        """
        Fetch the latest data from the database.

        Returns:
            datetime: The timestamp of the latest data in the database.
        """
        with self.connection.cursor() as cur:
            cur.execute("SELECT MAX(time) FROM sensor_data WHERE elec_prices_metric IS NOT NULL;")
            latest_time = cur.fetchone()[0]
        logger.info(f"Latest data fetched from database: {latest_time}.")
        return latest_time

    def _predict_historical_data(self, model_infos):
        """
        Predict and insert data for all historical records using the specified model.

        Args:
            model_infos (dict): Information about the model to use for prediction.
        """
        start_date = datetime.strptime(model_infos["periode"]["start_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        current_time = start_date + timedelta(hours=102)
        column_name = model_infos["column_name"]

        if model_infos["periode"]["end_date"]:
            latest_time = datetime.strptime(model_infos["periode"]["end_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            latest_time = self._fetch_latest_data().replace(tzinfo=timezone.utc)

        current_time = current_time.replace(hour=latest_time.hour, minute=latest_time.minute, second=latest_time.second)

        while current_time <= (latest_time - timedelta(hours=24)):
            last_100_hours, _ = self._get_last_100_hours(up_to_time=current_time)
            if last_100_hours is not None:
                predictions_df = self._predict_next_24_hours(last_100_hours, current_time,model_infos)
                self._upsert_data(predictions_df,column_name)
            current_time += timedelta(hours=24)
        logger.info(f"Historical data prediction of {column_name} completed.")
    def fetch_and_insert_new_data(self):
        """
        Continuously fetch new data from the API and insert it into the database.

        This method runs in an infinite loop, fetching new data every hour,
        inserting it into the database, and making predictions using the available models.
        """
        while True:
            # Récupérer la dernière donnée disponible via l'API
            latest_time = self._fetch_latest_data_API()

            # Définir la fenêtre de temps à interroger (100 dernières heures)
            start_ts = int((latest_time - timedelta(hours=100)).timestamp() * 1000)
            end_ts = int(latest_time.timestamp() * 1000)

            # Récupérer les nouvelles données
            data = self._fetch_data(start_ts, end_ts)

            if data:
                # Conversion en DataFrame et préparation des données
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df = df.rename(columns={"value": "elec_prices_metric"})

                # Insertion des nouvelles données
                self._upsert_data(df, "elec_prices_metric")
                logger.info("New data fetched and inserted in BDD.")

                # Récupérer les données des dernières 100 heures
                last_hour_data, last_hour_time = self._get_last_100_hours()

                if last_hour_data is not None:
                    # Boucler sur les modèles pour générer les prédictions
                    for model_key in ["model_4", "model_5", "model_6"]:
                        model_info = self.config.model_infos[model_key]
                        predictions_df = self._predict_next_24_hours(last_hour_data, last_hour_time, model_info)
                        
                        # Insertion des prédictions
                        self._upsert_data(predictions_df, model_info["column_name"])
                        logger.info(f"Predictions made for {model_info['column_name']}.")

            time.sleep(3600)  # 1 heure

    def run(self):
        """
        Main method to run the data collection and prediction process.

        This method initiates the continuous process of fetching and inserting new data.
        """
        self.fetch_and_insert_new_data()
        print("ok")

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.run()