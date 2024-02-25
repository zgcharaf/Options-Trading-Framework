from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

class IVTermStructure:
    def __init__(self, df):
        """
        Initialize the IVTermStructure with a DataFrame that includes option IDs.
        Assumes df includes 'QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE', 'C_IV', and 'Option_ID' columns.
        """
        self.df = df
        self.df['QUOTE_DATE'] = pd.to_datetime(self.df['QUOTE_DATE'])
        self.df['EXPIRE_DATE'] = pd.to_datetime(self.df['EXPIRE_DATE'])
        self.df.sort_values(by=['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE'], inplace=True)

    def extract_term_structure(self, group):
        """
        Extracts the IV term structure and the IDs of the options used for each group.
        """
        term_structure = group[['EXPIRE_DATE', 'C_IV']].set_index('EXPIRE_DATE').T.to_dict('records')[0]
        option_ids = group['Option_Contract_ID'].tolist()  # Collect all option IDs in the group
        return term_structure, option_ids

    def get_iv_term_structure(self, max_workers=4):
        """
        Calculates the IV term structure using multithreading and includes option IDs.
        """
        groups = list(self.df.groupby(['QUOTE_DATE', 'STRIKE']))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_term_structure, group): (quote_date, strike) 
                       for (quote_date, strike), group in groups}
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='Calculating IV Term Structures'):
                quote_date, strike = futures[future]
                term_structure, option_ids = future.result()
                results.append({
                    'QUOTE_DATE': quote_date, 
                    'STRIKE': strike, 
                    'IV_Term_Structure': term_structure,
                    'Option_IDs': option_ids
                })
        
        term_structures_df = pd.DataFrame(results)
        return term_structures_df

