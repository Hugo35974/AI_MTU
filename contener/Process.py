import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import requests
from joblib import dump, load
from psycopg2 import sql

from src.Pretreatment.ConfigLoader import ConfigLoader

Main_path = Path(__file__).parents[0]

class DataCollector:
    def __init__(self):
        self.configbdd = ConfigLoader().bdd
        self.connection = None
        self.api_url = "https://thingsboard.tec-gateway.com/unity/timeseries"
        self.api_url_latest = "https://thingsboard.tec-gateway.com/unity/latest"
        self.model_path = os.path.join(Main_path, "composite_model.pkl")
        self.docker_path = "docker-compose.yml"
        self.run_docker_compose(['up', '-d'])
        self._init_connection()
        self._create_table()
        self.inject_historical_data()
        self._predict_historical_data()

    def _init_connection(self):
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(
                    dbname=self.configbdd["postgre"]["dbname"],
                    user=self.configbdd["postgre"]["user"],
                    password=self.configbdd["postgre"]["password"],
                    host=self.configbdd["postgre"]["host"],
                    port=self.configbdd["postgre"]["port"]
                )
            except psycopg2.DatabaseError as e:
                print(f"Database connection error: {e}")
                sys.exit(1)

    def run_docker_compose(self,command):
        """
        Exécute une commande docker-compose.

        :param command: Liste des arguments pour la commande docker-compose.
        """
        try:
            # Exécuter la commande docker-compose
            result = subprocess.run(
                ['docker-compose', '-f', self.docker_path] + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            # Afficher la sortie standard
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # Afficher les erreurs en cas d'échec
            print(f"Erreur lors de l'exécution de la commande: {e.stderr}")


    def _close_connection(self):
        if self.connection is not None and not self.connection.closed:
            self.connection.close()

    def _create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sensor_data (
            time TIMESTAMP PRIMARY KEY,
            value FLOAT,
            elec_prices FLOAT
        );
        """
        with self.connection.cursor() as cur:
            cur.execute(create_table_query)
            self.connection.commit()

    def _fetch_data(self, start_ts, end_ts):
        params = {
            'startTs': start_ts,
            'endTs': end_ts
        }
        response = requests.get(self.api_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None

    def _upsert_data(self, df):
        with self.connection.cursor() as cur:
            for idx, row in df.iterrows():
                cur.execute("""
                    INSERT INTO sensor_data (time, elec_prices_metric, elec_prices)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (time)
                    DO UPDATE SET 
                        elec_prices_metric = COALESCE(sensor_data.elec_prices_metric, EXCLUDED.elec_prices_metric),
                        elec_prices = COALESCE(sensor_data.elec_prices, EXCLUDED.elec_prices);
                """, (row['time'], row['value'], row['elec_prices']))
        self.connection.commit()

    def date_to_timestamp(self, date_str):
        date_obj = datetime.strptime(date_str, "%Y/%m/%d")
        timestamp = int(date_obj.timestamp())
        timestamp_ms = timestamp * 1000
        return timestamp_ms

    def inject_historical_data(self):
        latest_data = self._fetch_latest_data()
        if latest_data:
            latest_time = pd.to_datetime(latest_data["price"][0]["ts"], unit='ms', utc=True)
            start_ts = self.date_to_timestamp("2024/01/01")
            end_ts = int(latest_time.timestamp() * 1000)  # End at the latest time available

            data = self._fetch_data(start_ts, end_ts)
            if data:
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df['elec_prices'] = None  # Initialize the prediction column
                self._upsert_data(df)

    def _get_last_100_hours(self, up_to_time=None):
        query = sql.SQL("""
        SELECT time, elec_prices_metric
        FROM sensor_data
        WHERE elec_prices_metric IS NOT NULL 
        {time_condition}
        ORDER BY time DESC
        LIMIT 100;
        """).format(time_condition=sql.SQL("AND time <= %s") if up_to_time else sql.SQL(""))
        with self.connection.cursor() as cur:
            cur.execute(query, (up_to_time,) if up_to_time else None)
            result = cur.fetchall()
            if len(result) < 100:
                print("Not enough non-null data to predict.")
                return None, None
            # Separate times and values
            times, values = zip(*result)
            return np.array(values), times[0]

    def _predict_next_24_hours(self, last_100_hours, start_time):
        model = load(self.model_path)

        time_tomorrow = start_time + timedelta(days=1)
        day_of_week = time_tomorrow.weekday()
        day_of_year = time_tomorrow.timetuple().tm_yday
        week_of_year = time_tomorrow.isocalendar()[1]

        date_features = np.array([day_of_week, day_of_year, week_of_year])
        x = np.concatenate((last_100_hours.ravel(), date_features))

        predictions = model.predict(x.reshape(1, -1))

        future_dates = [start_time + timedelta(hours=i) for i in range(24)]

        return pd.DataFrame({
            'time': future_dates,
            'elec_prices': predictions.flatten(),
            'value': None  # Keep 'value' as None for predicted rows
        })

    def _fetch_latest_data(self):
        response = requests.get(self.api_url_latest)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch latest data: {response.status_code}")
            return None

    def _predict_historical_data(self):
        """
        Predict and insert data for all historical records using the model.
        """
        # Start predictions 100 hours after the first recorded data
        with self.connection.cursor() as cur:
            cur.execute("SELECT MIN(time) FROM sensor_data")
            first_record_time = cur.fetchone()[0]
        
        current_time = first_record_time + timedelta(hours=100)

        while current_time <= datetime.now(timezone.utc):
            last_100_hours, _ = self._get_last_100_hours(up_to_time=current_time)
            if last_100_hours is not None:
                predictions_df = self._predict_next_24_hours(last_100_hours, current_time)
                self._upsert_data(predictions_df)
            current_time += timedelta(hours=24)  # Move to the next 24-hour period
    def _retrain_model(self):
        # Récupérer les données des 12 derniers mois
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=365)

        query = sql.SQL("""
        SELECT elec_prices_metric, elec_prices
        FROM sensor_data
        WHERE time BETWEEN %s AND %s
        """)
        with self.connection.cursor() as cur:
            cur.execute(query, (start_time, end_time))
            data = cur.fetchall()

        if len(data) < 100:
            print("Pas assez de données pour réentraîner le modèle.")
            return

        X_train = np.array([d[0] for d in data])
        y_train = np.array([d[1] for d in data])

        model = load(self.model_path)
        model.fit(X_train, y_train)

        # Sauvegarder le nouveau modèle
        dump(model, self.model_path)
        print("Modèle réentraîné et sauvegardé.")

    def fetch_and_insert_new_data(self):
        while True:
            self._init_connection()  # Ensure the connection is open
            latest_data = self._fetch_latest_data()
            latest_time = pd.to_datetime(latest_data["price"][0]["ts"], unit='ms', utc=True)

            start_ts = int((latest_time - timedelta(hours=100)).timestamp() *1000)
            end_ts = int(latest_time.timestamp() * 1000)
            data = self._fetch_data(start_ts,end_ts)

            if data:
                df = pd.DataFrame(data['price'], columns=['ts', 'value'])
                df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df['elec_prices'] = None  # Initialize the prediction column
                self._upsert_data(df)

                # Use the last available hour as the starting point for predictions
                last_hour_data, last_hour_time = self._get_last_100_hours()
                if last_hour_data is not None:
                    predictions_df = self._predict_next_24_hours(last_hour_data, last_hour_time)
                    self._upsert_data(predictions_df)

            self._close_connection()  # Close the connection after processing
            time.sleep(3600)

    def run(self):
        while True:
            self.fetch_and_insert_new_data()

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.run()
