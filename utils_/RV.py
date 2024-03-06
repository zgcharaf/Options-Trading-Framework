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
        # Ensure the DataFrame is sorted by QUOTE_DATE to correctly shift the IV values
        df = self.df.sort_values('QUOTE_DATE')
        
        # Shift the IV column to get the IV at t-1
        df['IV_t_minus_one'] = df.groupby(['CONTRACT_ID'])['IV'].shift(1)
        
        # Merge the RV calculated DataFrame to include RV for each date
        df_with_rv = self.calculate_RV()
        
        # Combine the RV and IV(t-1) in the same DataFrame
        df_combined = df_with_rv[['QUOTE_DATE', 'CONTRACT_ID', 'rolling_yearly_vol', 'IV_t_minus_one']].copy()
        
        # Optionally, filter out rows where IV_t_minus_one is NaN, which happens for the first date of each contract
        df_combined.dropna(subset=['IV_t_minus_one'], inplace=True)
        
        return df_combined
