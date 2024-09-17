import json
import os
import sys
from pathlib import Path

# Define project paths
PROJECT_PATH = Path(__file__).parents[2]
MAIN_PATH = Path(__file__).parents[0]
SRC_PATH = Path(__file__).resolve().parents[2] 

sys.path.append(PROJECT_PATH)
sys.path.append(MAIN_PATH)
sys.path.append(str(SRC_PATH))
from src.Pretreatment.DataProcessor import DataProcessor


class ConfigLoader:
    def __init__(self):
        self.config = self.load_config()
        self.date_config = self.extract_date()
        self.variables_config = self.extract_variables()
        self.model_config = self.extract_model()
        self.file_imports = self.extract_file_imports()
        self.transformations = self.extract_transformations()
        self.mode = self.extract_mode()
        self.bdd = self.extract_bdd_setting()
        self.models = self.extract_model_list()
        self.api = self.extract_api()
        self.model_infos = self.extract_model_infos()

    def load_config(self):
        conf_path = os.path.join(PROJECT_PATH, 'config.json')
        with open(conf_path) as file:
            config = json.load(file)
        return config

    def extract_date(self):
        return self.config.get('date', {})

    def extract_variables(self):
        return self.config.get('variables', {})

    def extract_model(self):
        return self.config.get('model', {})

    def extract_file_imports(self):
        return self.config.get('file_imports', {}).get('files', [])

    def launch_data_processor(self):
        processor = DataProcessor(self.file_imports[0], self.file_imports[1:])
        return processor.get_final_data()

    def extract_transformations(self):
        return self.config.get('transformations', {})

    def extract_mode(self):
        return self.config.get('mode', [])

    def extract_bdd_setting(self):
        return self.config.get('bdd', [])

    def extract_model_list(self):
        return self.config.get('model', [])

    def extract_api(self):
        return self.config.get('api', [])

    def extract_model_infos(self):
        return self.config.get('models_infos', [])
