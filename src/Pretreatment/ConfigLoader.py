import json
import os
import sys
from pathlib import Path

Project_Path = Path(__file__).parents[2]
Main_Path = Path(__file__).parents[0]

sys.path.append(Project_Path)
sys.path.append(Main_Path)

from src.Pretreatment.DataProcessor import DataProcessor


class ConfigLoader():
    def __init__(self):
        self.config = self.loadconfig()
        self.date_config = self.extractdate()
        self.variables_config = self.extractvariables()
        self.model_config = self.extractmodel()
        self.file_imports = self.extractfileimports()
        self.transformations = self.extracttransformations()
        self.mode = self.extractmode()
        self.bdd = self.extractBddSetting()
        self.models = self.extractModel()
        self.api = self.extractAPI()

    def loadconfig(self):
        conf_path = os.path.join(Project_Path,'config.json')
        with open(conf_path) as file:
            config = json.load(file)
        return config 
    
    def extractdate(self):
        return self.config.get('date', {})

    def extractvariables(self):
        return self.config.get('variables', {})

    def extractmodel(self):
        return self.config.get('model', {})

    def extractfileimports(self):
        return self.config.get('file_imports', {}).get('files', [])
    
    def launchdataprocessor(self):
        processor = DataProcessor(self.file_imports[0],self.file_imports[1:])
        return processor.get_final_data()
    
    def extracttransformations(self):
        return self.config.get('transformations', {})
    
    def extractmode(self):
        return self.config.get('mode',[])
    
    def extractBddSetting(self):
        return self.config.get('bdd',[])
    
    def extractModel(self):
        return self.config.get('model',[])
    
    def extractAPI(self):
        return self.config.get('api',[])