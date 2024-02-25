
import os
import json
from datetime import datetime, timedelta
import pandas as pd 
import numpy as np 
from fredapi import Fred
import requests
APIKEY = '230970faf44ea208229d77dff9f995f3'




def current_time_formatted(format="%Y-%m-%d %H:%M:%S"):
    """
    Returns the current datetime formatted as a string.
    
    :param format: The datetime format string.
    :return: The formatted datetime string.
    """
    return datetime.now().strftime(format)

def days_from_now(days, format="%Y-%m-%d"):
    """
    Returns a date that is a certain number of days from today, formatted as a string.
    
    :param days: Number of days from today.
    :param format: The date format string.
    :return: The formatted date string.
    """
    
    future_date = datetime.now() + timedelta(days=days)
    return future_date.strftime



def read_data_csv(paths=[]):
    """
    Reads data from a list of CSV file paths and concatenates them into a single DataFrame.

    :param paths: List of strings, where each string is a path to a CSV file.
    :return: A pandas DataFrame containing the concatenated data from all CSV files.
    """
    df = pd.DataFrame()
    for path in paths:
        dfi = pd.read_csv(path)
        df = pd.concat([df, dfi], ignore_index=True)
    return df


def clean_and_transform_market_data(df):
    """
    Cleans and transforms market data DataFrame.
    
    :param df: pandas DataFrame containing market data.
    :return: Transformed pandas DataFrame.
    """
    df.columns = [col.replace('[', '').replace(']', '').strip() for col in df.columns]
    # Replace ' ' with np.nan
    df = df.replace(' ', np.nan)
    
    # Sorting the DataFrame by QUOTE_READTIME
    df = df.sort_values(by='QUOTE_READTIME')
    
    # Converting columns to datetime
    df['QUOTE_READTIME'] = pd.to_datetime(df['QUOTE_READTIME'])
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
    
    # Converting columns to float
    float_columns = ['C_IV', 'P_IV', 'C_BID', 'P_BID', 'P_ASK', 'C_ASK', 'C_DELTA', 'P_DELTA', 'C_GAMMA', 'P_GAMMA', 'C_VEGA', 'C_RHO', 'STRIKE', 'UNDERLYING_LAST', 'C_THETA', 'C_VOLUME', 'P_VOLUME']
    for col in float_columns:
        df[col] = df[col].astype('float')
    
    # Calculating bid-ask spreads
    df['C_BIDASKSPREAD'] = abs(df['C_ASK'] - df['C_BID'])
    df['P_BIDASKSPREAD'] = abs(df['P_ASK'] - df['P_BID'])
    
    # Calculating mid prices
    df['C_MIDPRICE'] = (df['C_BID'] + df['C_ASK']) / 2
    df['P_MIDPRICE'] = (df['P_BID'] + df['P_ASK']) / 2
    
    # Calculating prices as percentages of the underlying last price
    df['C_AS_PERCT_OF_UNDER'] = df['C_MIDPRICE'] / df['UNDERLYING_LAST']
    df['P_AS_PERCT_OF_UNDER'] = df['P_MIDPRICE'] / df['UNDERLYING_LAST']
    
    # Cleaning column names
    
    
    return df



def get_from_fred_with_fredapi( series_ids, start_date=None, end_date=None):
    
    """
    Fetches economic data from the FRED API using fredapi for given series IDs, with optional date range.

    :param api_key: String, FRED API key for authenticating requests.
    :param series_ids: List of strings, FRED series IDs to fetch data for.
    :param start_date: String, start date for data retrieval in 'YYYY-MM-DD' format. Optional.
    :param end_date: String, end date for data retrieval in 'YYYY-MM-DD' format. Optional.
    :return: A pandas DataFrame containing the data for all requested series.
    """
    fred = Fred(api_key=APIKEY)
    all_data = []

    for series_id in series_ids:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = pd.DataFrame(data, columns=[series_id])
        all_data.append(df)

    # Combine all series into a single DataFrame
    combined_data = pd.concat(all_data, axis=1)
    return combined_data

def process_and_merge_data( csv_paths, series_ids, start_date, end_date):
    """
    Processes market data from CSV files, fetches and interpolates risk-free rate data from FRED,
    and merges them on a common 'QUOTE_date'.

    :param api_key: String, FRED API key for authenticating requests.
    :param csv_paths: List of strings, paths to the CSV files containing market data.
    :param series_ids: List of strings, FRED series IDs to fetch data for.
    :param start_date: String, start date for data retrieval in 'YYYY-MM-DD' format.
    :param end_date: String, end date for data retrieval in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame containing the merged market and risk-free rate data.
    """
    # Step 1: Read and concatenate market data from CSV files
    df = read_data_csv(csv_paths)
    
    # Step 2: Clean and transform the market data
    df = clean_and_transform_market_data(df)
    df['QUOTE_date'] = pd.to_datetime(df['QUOTE_READTIME'],format='%Y-%m-%d') 
    df['QUOTE_date'] = df['QUOTE_READTIME'].dt.normalize()
    df['C_VOLUME']=  df['C_VOLUME'].astype('float')
    df['Option_Contract_ID'] = ("OP" + df['STRIKE'].astype(str) + '-'+ df['EXPIRE_DATE'].dt.month.astype(str).str.zfill(2) +'-'+  df['EXPIRE_DATE'].dt.year.astype(str))
    # Step 3: Fetch risk-free rate data from FRED
    rf_data = get_from_fred_with_fredapi(  series_ids=series_ids, start_date=start_date, end_date=end_date)
    
    # Additional steps to process rf_data as described
    rf_data = rf_data.reset_index().rename(columns={"index": 'QUOTE_date'})
    rf_data['QUOTE_date'] = pd.to_datetime(rf_data['QUOTE_date'], format='%Y-%m-%d')
    rf_data_daily = rf_data.set_index('QUOTE_date').resample('D').interpolate('linear').reset_index()
    
    # Step 4: Merge market data with interpolated risk-free rate data
    dfx = df.merge(rf_data_daily, how='left', on='QUOTE_date')
    
    return dfx
    