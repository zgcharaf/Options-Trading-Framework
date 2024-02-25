from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd

class IVStructureAnalysis:
    def __init__(self, term_structures_df):
        self.term_structures_df = term_structures_df

    def calculate_forward_volatility(self, short_term_iv, long_term_iv, short_term_days, long_term_days):
        """
        Calculate forward volatility between two periods.
        """
        numerator = (long_term_iv ** 2 * long_term_days) - (short_term_iv ** 2 * short_term_days)
        denominator = long_term_days - short_term_days
        forward_vol = np.sqrt(numerator / denominator)
        return forward_vol

    def analyze_structure_single(self, row):
        quote_date = row['QUOTE_DATE']
        iv_term_structure = row['IV_Term_Structure']
        option_ids = row['Option_IDs']

        # Ensure there are at least two points in the IV term structure for comparison
        if len(iv_term_structure) < 10:
            return []

        # Sort the term structure by expiration date
        sorted_term_structure = sorted(iv_term_structure.items(), key=lambda x: x[0])
        short_term_iv, long_term_iv = sorted_term_structure[0][1], sorted_term_structure[-1][1]
        short_term_days = (sorted_term_structure[0][0] - quote_date).days
        long_term_days = (sorted_term_structure[-1][0] - quote_date).days
        
        # Calculate forward volatility
        forward_vol = self.calculate_forward_volatility(short_term_iv, long_term_iv, short_term_days, long_term_days)

        decisions = []

        # Assuming option_ids is structured with ids and corresponding C_BID and C_ASK values
        # Here, we need a method to map option_ids to their C_BID and C_ASK. Adjust as per your data structure.
        for option_id in option_ids:
    # Determine the decision and price based on IV comparison
            decision = 'Sell' if short_term_iv < long_term_iv else 'Buy'
            price_column = 'C_BID' if decision == 'Sell' else 'C_ASK'
    
    # Filter the DataFrame once per option_id to avoid repetition
            option_data = df[(df['Option_Contract_ID'] == option_id) & (df['QUOTE_DATE'] == quote_date)]
    
            if not option_data.empty:
                decisions.append({
            'DATE': quote_date,
            'DECISION': decision,
            'CONTRACT_ID': option_id,
            'PRICE': option_data[price_column].values[0],
            'QUANTITY': 1,
            'DTE': option_data['DTE'].values[0],
            'C_DELTA': option_data['C_DELTA'].values[0],
            'C_GAMMA': option_data['C_GAMMA'].values[0],
            'C_VEGA': option_data['C_VEGA'].values[0],
            'EXPIRE_DATE': option_data['EXPIRE_DATE'].values[0],
            'UNDERLYING': option_data['UNDERLYING_LAST_x'].values[0],
        })


        return decisions

    def analyze_structure(self, max_workers=4):
        all_decisions = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.analyze_structure_single, row) for _, row in self.term_structures_df.iterrows()]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Analyzing IV Structures'):
                result = future.result()
                if result:
                    all_decisions.extend(result)

        return pd.DataFrame(all_decisions)
