import pandas as pd 
from Model import XGB
from Pretreatment.DataProcessor import DataProcessor
from Tools.tools import shifting,plot_prediction_results,eval_metrics
from pathlib import Path
import os
import json

Project_Path = Path(__file__).parents[1]

class ConfigLoader():
    def __init__(self) -> None:
        self.config = self.loadconfig()
        self.date_config = self.extractdate()
        self.variables_config = self.extractvariables()
        self.model_config = self.extractmodel()
        self.file_imports = self.extractfileimports()
        self.transformations = self.extracttransformations()

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
