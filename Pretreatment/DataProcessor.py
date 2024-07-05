import pandas as pd
from pathlib import Path
import os 

Project_Path = Path(__file__).parents[1]
Data_Path = os.path.join(Project_Path,"Data")

class DataProcessor:
    def __init__(self, prices_file, *additional_files):
        path_prices_file = os.path.join(Data_Path,prices_file)
        self.data = pd.read_csv(path_prices_file, index_col="applicable_date")
        self.data.index = pd.to_datetime(self.data.index, format='%d/%m/%Y %H:%M')
        self.data = pd.DataFrame(self.data['ROI-DA'])
        
        self.additional_data = []
        for file in additional_files[0]:
            path_file = os.path.join(Data_Path,file)
            df = pd.read_csv(path_file, index_col=0)
            df.index = pd.to_datetime(df.index, format='%m/%d/%Y')
            date_debut = df.index.min()
            date_fin = df.index.max() + pd.Timedelta(days=1)
            index_horaire = pd.date_range(date_debut, date_fin, freq='H')
            df = df.reindex(index=index_horaire, method='ffill')
            self.additional_data.append(df)
        
        self._process_data()

    def _process_data(self):
        self.df_final = self.data.copy()
        for df in self.additional_data:
            self.df_final = pd.merge(self.df_final, df, left_index=True, right_index=True, how='left')

    def get_final_data(self):
        return self.df_final
