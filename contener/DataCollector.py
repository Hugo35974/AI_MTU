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

from src.Pretreatment.ConfigLoader import ConfigLoader
from src.Runner.Runner import Run

Main_path = Path(__file__).parents[0]
Project_path = Path(__file__).parents[1]
Timeseries_path = os.path.join(Project_path, "data", "timeseries")

class DataCollector:
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
        self._predict_historical_data(self.config.model_infos["model_1"])
        # self._retrain_model(self.config.model_infos["model_2"])
        self._predict_historical_data(self.config.model_infos["model_2"])

    def _init_connection(self):
        if not self.connection or self.connection.closed:
            try:
                self.connection = psycopg2.connect(
                    dbname=self.config.bdd["dbname"],
                    user=self.config.bdd["user"],
                    password=self.config.bdd["password"],
                    host=self.config.bdd["host"],
                    port=self.config.bdd["port"]
                )
                logger.info("Database connection established successfully.")
            except psycopg2.DatabaseError as e:
                logger.error(f"Database connection error: {e}")
                sys.exit(1)

    def _close_connection(self):
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed.")

    def _create_tables(self):
        queries = [
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                time TIMESTAMP PRIMARY KEY,
                elec_prices_metric FLOAT,
                elec_prices FLOAT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP,
                level VARCHAR(10),
                message TEXT
            );
            """
        ]
        with self.connection.cursor() as cur:
            for query in queries:
                cur.execute(query)
            self.connection.commit()
        logger.info("Tables created or verified successfully.")

    def _log_to_db(self, record):
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
        params = {'startTs': start_ts, 'endTs': end_ts}
        response = requests.get(self.api_url, params=params)
        if response.status_code == 200:
            logger.info(f"Data fetched successfully from API for period: {start_ts} to {end_ts}.")
            return response.json()
        logger.error(f"Failed to fetch data from API. Status code: {response.status_code}")
        return None

    def _upsert_data(self, df):
        with self.connection.cursor() as cur:
            for idx, row in df.iterrows():
                cur.execute("""
                    INSERT INTO sensor_data (time, elec_prices_metric, elec_prices)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (time)
                    DO UPDATE SET elec_prices = COALESCE(EXCLUDED.elec_prices, sensor_data.elec_prices);
                """, (row['time'], row['value'], row['elec_prices']))
        self.connection.commit()

    def date_to_timestamp(self, date_str):
        return int(datetime.strptime(date_str, "%Y/%m/%d").timestamp()) * 1000

    def inject_historical_data(self):
        try:
            latest_data = self._fetch_latest_data_API()
            start_ts = self.date_to_timestamp("2014/01/01")
            end_ts = int(latest_data.timestamp() * 1000)

            data = self._fetch_data(start_ts, end_ts)
            if data:
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df['elec_prices'] = None  
                self._upsert_data(df)
                logger.info("API historical data injected successfully.")

            csv_path = os.path.join(Timeseries_path, "Electricity_Prices_Ireland.csv")
            df = pd.read_csv(csv_path)
            df = df.rename(columns={"elec_prices": "value", "applicable_date": "time"})
            df['time'] = pd.to_datetime(df['time'])
            df = df.dropna()
            df['elec_prices'] = None
            self._upsert_data(df)
            logger.info("CSV data injected successfully.")

        except Exception as e:
            logger.error(f"An error occurred while injecting historical data: {e}")

    def _get_last_100_hours(self, up_to_time=None):
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
        model = load(os.path.join(Main_path, model_infos["path"]))
        date_features = np.array([
            start_time.weekday(), 
            start_time.timetuple().tm_yday,
            start_time.isocalendar()[1],
            start_time.month
        ])
        x = np.concatenate((last_100_hours.ravel(), date_features))
        predictions = model.predict(x.reshape(1, -1))

        future_dates = [start_time + timedelta(hours=i) for i in range(1, 25)]
        return pd.DataFrame({
            'time': future_dates,
            'elec_prices': predictions.flatten(),
            'value': None
        })

    def _fetch_latest_data_API(self):
        response = requests.get(self.api_url_latest)
        if response.status_code == 200:
            latest_data = response.json()
            logger.info("Latest data fetched from API successfully.")
            return pd.to_datetime(latest_data["price"][0]["ts"], unit='ms', utc=True)
        logger.error(f"Failed to fetch latest data from API. Status code: {response.status_code}")
        return None

    def _fetch_latest_data(self):
        with self.connection.cursor() as cur:
            cur.execute("SELECT MAX(time) FROM sensor_data;")
            latest_time = cur.fetchone()[0]
        logger.info(f"Latest data fetched from database: {latest_time}.")
        return latest_time

    def _predict_historical_data(self, model_infos):
        """
        Predict and insert data for all historical records using the model.
        """
        start_date = datetime.strptime(model_infos["periode"]["start_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        current_time = start_date + timedelta(hours=102)

        if model_infos["periode"]["end_date"]:
            latest_time = datetime.strptime(model_infos["periode"]["end_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            latest_time = self._fetch_latest_data().replace(tzinfo=timezone.utc)

        current_time = current_time.replace(hour=latest_time.hour, minute=latest_time.minute, second=latest_time.second)

        while current_time <= (latest_time - timedelta(hours=24)):
            last_100_hours, _ = self._get_last_100_hours(up_to_time=current_time)
            if last_100_hours is not None:
                predictions_df = self._predict_next_24_hours(last_100_hours, current_time,model_infos)
                self._upsert_data(predictions_df)
            current_time += timedelta(hours=24)
        logger.info("Historical data prediction completed.")

    def fetch_and_insert_new_data(self):
        while True:
            self._init_connection()
            latest_time = self._fetch_latest_data_API()

            start_ts = int((latest_time - timedelta(hours=100)).timestamp() * 1000)
            end_ts = int(latest_time.timestamp() * 1000)
            data = self._fetch_data(start_ts, end_ts)

            if data:
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df['elec_prices'] = None
                self._upsert_data(df)

                last_hour_data, last_hour_time = self._get_last_100_hours()
                if last_hour_data is not None:
                    predictions_df = self._predict_next_24_hours(last_hour_data, last_hour_time, self.config.model_infos["model_2"])
                    self._upsert_data(predictions_df)

            self._close_connection()
            logger.info("New data fetched and predictions made.")
            time.sleep(3600)

    def run(self):
        self.fetch_and_insert_new_data()

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.run()