import pandas as pd 
import numpy as np
from utils_ import utilss
from utils_ import ImpliedVar
from models.Core import Core
import matplotlib.pyplot as plt 
import warnings 
from utils_ import IVTermStructure
from utils_ import IVStructureAnalysis
from utils_ import RV
import tqdm
from models import FiniteDifference
warnings.filterwarnings('ignore')
from transactions import transactions
from portfolio import portfolio


def main():
    dfx = utilss.process_and_merge_data(['data/aapl_2016_2020.csv'], ['TB3MS'], start_date='2016-01-01', end_date='2023-04-30')
    dfx['C_VOLUME'] =  dfx['C_VOLUME'].astype('float')
    dfx['P_VOLUME'] =  dfx['P_VOLUME'].astype('float')
    dfx=dfx.dropna()
    df_sub=dfx.iloc[:10000,  :]
    dfx=df_sub
    opml_instance = Core(dfx)
    df=opml_instance.add_greeks_to_df()
    df=opml_instance.add_bs_call_price()
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], format='%Y-%m-%d')
    df['Option_Contract_ID'] = ("OP" + df['STRIKE'].astype(str) + '-'+df['EXPIRE_DATE'].dt.day.astype(str).str.zfill(2) +'-'+df['EXPIRE_DATE'].dt.month.astype(str).str.zfill(2) +'-'+  df['EXPIRE_DATE'].dt.year.astype(str))

    iv_term_structure = IVTermStructure.IVTermStructure(df)
    term_structures_df = iv_term_structure.get_iv_term_structure_modified()
    processed_df = iv_term_structure.prepare_training_df()
    tensors= iv_term_structure.generate_matrices(term_structures_df)
    breakpoint()
    print(tensors)

    
    
if __name__ == "__main__":
    main()

