import numpy as np
import pandas as pd

class RV(): 
    def __init__(self, df): 
        self.df = df

    def calculate_RV(self): 
        df = self.df
        x = df.groupby('QUOTE_DATE')['UNDERLYING_LAST'].max().reset_index()
        x['log_price'] = np.log(x['UNDERLYING_LAST'])
        x['day_log_return'] = x['log_price'].diff()
        x['rolling_yearly_vol'] = x['day_log_return'].rolling(window=252, min_periods=1).std() * np.sqrt(252)
        df = df.merge(x[['QUOTE_DATE', 'rolling_yearly_vol']], on='QUOTE_DATE', how='left')
        return df 

    def RV_t_IV_t_minus_one(self):
        pass 
