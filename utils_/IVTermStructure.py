from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import pandas as pd
from ast import literal_eval
import json

class IVTermStructure:
    def __init__(self, df):
        self.df = df
        self.df['QUOTE_DATE'] = pd.to_datetime(self.df['QUOTE_DATE'])
        self.df['EXPIRE_DATE'] = pd.to_datetime(self.df['EXPIRE_DATE'])
        self.df.sort_values(by=['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE'], inplace=True)
        self.training_df = pd.DataFrame()
        self.tensors = []
        self.processed_data = pd.DataFrame()


    def extract_term_structure(self, group):
   
        term_structure = group[['EXPIRE_DATE', 'C_IV']].set_index('EXPIRE_DATE').T.to_dict('records')[0]
        option_ids = group['Option_Contract_ID'].tolist()  # Ensure this column name matches your DataFrame

        option_deltas = {}
        option_vegas = {}
        option_gammas = {}
        option_rhos = {}
        option_thetas = {}
        options_DTE =  {} 
        Strike_DIS =  {} 

        for _, row in group.iterrows():
            contract_id = row['Option_Contract_ID']
        # Map Greeks to each contract ID
            option_deltas[contract_id] = row['C_DELTA']
            option_vegas[contract_id] = row['C_VEGA']
            option_gammas[contract_id] = row['C_GAMMA']
            option_rhos[contract_id] = row['C_RHO']
            option_thetas[contract_id] = row['C_THETA']
            options_DTE[contract_id]= row['DTE']
            Strike_DIS[contract_id]= row['STRIKE_DISTANCE']

        return term_structure, option_ids, option_deltas, option_vegas, option_gammas, option_rhos, option_thetas, options_DTE, Strike_DIS


    def get_iv_term_structure_modified(self, max_workers=4):
        groups = list(self.df.groupby(['QUOTE_DATE', 'STRIKE']))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_term_structure, group): (quote_date, strike) 
                       for (quote_date, strike), group in groups}

            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='Calculating IV Term Structures'):
                quote_date, strike = futures[future]
                term_structure, option_ids, deltas, vegas, gammas, rhos , thetas, DTE, Strike_DIS= future.result()
                # Check if the term structure has at least 10 different maturity rates
                if len(term_structure) >= 10:
                    results.append({
                        'QUOTE_DATE': quote_date,
                        'STRIKE': strike,
                        'IV_Term_Structure': term_structure,
                        'Option_IDs': option_ids,
                        'DELTA': deltas,
                        'VEGA': vegas,
                        'GAMMA': gammas,
                        'RHO': rhos,
                        'THETA': thetas, 
                        'DTE': DTE, 
                        'STRIKE_DIS': Strike_DIS
                    })

        term_structures_df = pd.DataFrame(results)
        self.training_df = term_structures_df
        return term_structures_df
        
   # def prepare_training_df(self, max_workers=4) : 
    #    df = self.training_df 
        # for every Quote_DATE 
        # True_Time is a vector of unique dates quote dates 
        # expiration dates is the vector of expiration dates for every K and for every quote_date 
        # options_contracts is the vector of contracts making the term structure for every K and for every strike 
        # greeks is a matrix for every K and for every quote date of all the greeks for the contracts, rows correspond the every options contract in the options_contracts vector
        # DTE is a vector of the DTE for every quote_date and every strike 

        # IV is the vector of is the vector implied volatilities for a given K and for a given quote date 
        
    def safe_eval(self, value):
        try:
            # Only apply eval if the value is a string
            if isinstance(value, str):
                return eval(value)
            else:
                return value
        except:
            return value
            
    def prepare_training_df(self):
        grouped = self.training_df.groupby(['QUOTE_DATE', 'STRIKE'])

        processed_data = []

        for (quote_date, strike), group in grouped:
            options_contracts_for_strike = group['Option_IDs'].explode().unique().tolist()

            # Adjustments for safely evaluating or directly using the data
            greeks_for_strike = {greek: group[greek].apply(self.safe_eval).tolist() for greek in ['DELTA', 'VEGA', 'GAMMA', 'RHO', 'THETA']}
            dte_for_strike = group['DTE'].apply(self.safe_eval).tolist()
            iv_for_strike = group['IV_Term_Structure'].apply(self.safe_eval).tolist()

            processed_group = {
                'quote_date': quote_date,
                'strike': strike,
                'option_contracts': options_contracts_for_strike,
                'greeks': greeks_for_strike,
                'dte': dte_for_strike,
                'iv': iv_for_strike,
            }
            processed_data.append(processed_group)
            self.processed_data = processed_data

        return pd.DataFrame(processed_data)


    @staticmethod
    def safe_convert_to_dict(column_value):
        """Safely convert a column value to a dictionary."""
        if isinstance(column_value, dict):
            return column_value
        try:
            # Attempt to directly use literal_eval if the format is correct
            return literal_eval(column_value)
        except ValueError:
            # If literal_eval fails, try converting with json.loads after replacing single quotes
            try:
                return json.loads(column_value.replace("'", '"'))
            except json.JSONDecodeError:
                return {}

    def generate_matrices(self, term_structures_df):
        tensors = []  # List to store the matrices generated for each row

        for index, row in term_structures_df.iterrows():
        # Convert stringified columns to dictionaries if necessary
            iv_structure = self.safe_convert_to_dict(row['IV_Term_Structure'])
            delta = self.safe_convert_to_dict(row['DELTA'])
            vega = self.safe_convert_to_dict(row['VEGA'])
            gamma = self.safe_convert_to_dict(row['GAMMA'])
            rho = self.safe_convert_to_dict(row['RHO'])
            theta = self.safe_convert_to_dict(row['THETA'])
            dte = self.safe_convert_to_dict(row['DTE'])

            option_ids = row['Option_IDs'] if isinstance(row['Option_IDs'], list) else literal_eval(row['Option_IDs'])

        # Calculate the number of unique maturities in the IV structure
            num_maturities = len(iv_structure)

        # Calculate the max length required for padding
            max_length = max(num_maturities, len(option_ids))  # Assuming option_ids define the length for Greeks and DTE

        # Initialize the matrix with 7 rows for IV, each Greek, and DTE
            matrix = np.full((7, max_length), np.nan)  # Adjusted to 7 rows

        # Fill the matrix with IV values and Greeks for each option_id
            matrix[0, :num_maturities] = list(iv_structure.values())  # IV values
            for i, option_id in enumerate(option_ids):
                matrix[1, i] = delta.get(option_id, np.nan)  # Delta values
                matrix[2, i] = vega.get(option_id, np.nan)  # Vega values
                matrix[3, i] = gamma.get(option_id, np.nan)  # Gamma values
                matrix[4, i] = rho.get(option_id, np.nan)  # Rho values
                matrix[5, i] = theta.get(option_id, np.nan)  # Theta values
                matrix[6, i] = dte.get(option_id, np.nan)  # DTE values

        # Append the constructed matrix to the list of tensors
            tensors.append(matrix)

    # Return the list of tensors after processing all rows
        return tensors

