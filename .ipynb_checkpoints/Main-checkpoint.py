import pandas as pd 
import numpy as np
from utils_ import utilss
from models.Core import Core
import matplotlib.pyplot as plt 
import warnings 
import tqdm
import os 

def main():
    dfx = utilss.process_and_merge_data(['Data/nvda_2020_2022.csv', ], ['TB3MS'], start_date='2016-01-01', end_date='2023-04-30')
    opml_instance = Core(dfx)
    df=opml_instance.add_greeks_to_df()
    df=opml_instance.add_bs_call_price()
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], format='%Y-%m-%d')
    out = 'outputs'
    df.to_csv(os.path.join(out,'Df_with_b&s.csv'))
    
if __name__ == "__main__":
    main()

