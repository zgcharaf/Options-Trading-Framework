import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class Core:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """Calculate Black-Scholes call option price."""
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def deduce_r_binary_search(S, K, T, sigma, C_ask, r_min=0, r_max=0.1, tol=1e-4):
        """Deduce the risk-free rate from the ask price using a binary search."""
        while r_min < r_max:
            r_guess = (r_min + r_max) / 2
            C_guess = OPML.black_scholes_call(S, K, T, r_guess, sigma)
            
            if np.abs(C_guess - C_ask) < tol:
                return r_guess
            elif C_guess < C_ask:
                r_min = r_guess
            else:
                r_max = r_guess
        
        return (r_min + r_max) / 2

    def add_deduced_r_column_parallel_with_progress(self):
        """Add a column for the deduced risk-free rate to the DataFrame using parallel processing with a progress bar."""
        rows = self.df.to_dict('records')
        pbar = tqdm(total=len(rows), desc="Deducing Risk-Free Rates")
        
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.deduce_r_for_row, row): row for row in rows}
            
            for future in as_completed(futures):
                _ = future.result()
                pbar.update(1)
        
        pbar.close()
        self.df['deduced_r'] = [future.result() for future in futures]

    @staticmethod
    def deduce_r_for_row(row):
        """Wrapper function to apply deduce_r_binary_search for each row."""
        return OPML.deduce_r_binary_search(
            S=row['UNDERLYING_LAST'], 
            K=row['STRIKE'], 
            T=row['DTE'] / 365, 
            sigma=row['C_IV'], 
            C_ask=row['C_ASK']
        )

    
    def  add_greeks_to_df(self):
        
        """Add Greeks to the DataFrame for call options."""
        df = self.df
        days_in_year = 365.0

        df['T'] = df['DTE'] / days_in_year
        df['r'] = df['TB3MS'] / 100

        df['d1'] = (np.log(df['UNDERLYING_LAST'] / df['STRIKE']) + (df['r'] + 0.5 * df['C_IV']**2) * df['T']) / (df['C_IV'] * np.sqrt(df['T']))
        df['d2'] = df['d1'] - df['C_IV'] * np.sqrt(df['T'])

        df['C_DELTA_BS'] = norm.cdf(df['d1'])
        df['C_GAMMA_BS'] = norm.pdf(df['d1']) / (df['UNDERLYING_LAST'] * df['C_IV'] * np.sqrt(df['T']))
        df['C_THETA_BS'] = (-df['UNDERLYING_LAST'] * norm.pdf(df['d1']) * df['C_IV'] / (2 * np.sqrt(df['T'])) - df['r'] * df['STRIKE'] * np.exp(-df['r'] * df['T']) * norm.cdf(df['d2']))
        df['C_VEGA_BS'] = df['UNDERLYING_LAST'] * norm.pdf(df['d1']) * np.sqrt(df['T'])
        df['C_RHO_BS'] = df['STRIKE'] * df['T'] * np.exp(-df['r'] * df['T']) * norm.cdf(df['d2'])
        df['C_RHO_BS']= df['C_RHO_BS']/100
        # Adjustments
        df['C_THETA_BS'] = df['C_THETA_BS'] / days_in_year
        df['C_VEGA_BS'] = df['C_VEGA_BS'] / 100

        self.df = df
        return df


    def df_to_tensor(self, index_col='quote_date', value_cols=None, sort_cols=None):
        """
        Convert the DataFrame into a tensor, where each matrix corresponds to one unique index_col value,
        ensuring the order of values in every vector follows the order in the DataFrame.

        Parameters:
        - index_col: The name of the column to use as the basis for grouping into matrices.
        - value_cols: List of column names to include as values in the tensor. If None, use all columns except index_col.
        - sort_cols: Columns to sort by before grouping, ensuring consistent order within each group.
        
        Returns:
        - A tensor where each matrix corresponds to one unique value of index_col, with consistent ordering.
        """
        df = self.df.copy()
        
        if value_cols is None:
            value_cols = df.columns.difference([index_col])
        
        # Ensure consistent ordering within each group by sorting the DataFrame
        if sort_cols is None:
            sort_cols = [index_col]
        df.sort_values(by=sort_cols, inplace=True)
        
        # Convert index_col to 'category' for efficient processing
        df[index_col] = df[index_col].astype('category')
        
        # Group by index_col, convert each group to a matrix
        matrices = [group[value_cols].to_numpy() for _, group in df.groupby(index_col)]
        
        # Determine max shape for padding
        max_rows = max(matrix.shape[0] for matrix in matrices)
        max_cols = len(value_cols)
        
        # Pad matrices to uniform shape
        padded_matrices = [np.pad(matrix, ((0, max_rows - matrix.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for matrix in matrices]
        
        # Stack matrices to form a tensor
        tensor = np.stack(padded_matrices)
        
        return tensor


    def df_to_tensor_x(self, index_col='quote_date', value_cols=None, sort_cols=None):
        """
        Convert the DataFrame into a tensor, where each matrix corresponds to one unique index_col value,
        ensuring the order of values in every vector follows the order in the DataFrame.

        Parameters:
        - index_col (str): The name of the column to use as the basis for grouping into matrices.
        - value_cols (list of str): List of column names to include as values in the tensor. If None, use all columns except index_col.
        - sort_cols (list of str): Columns to sort by before grouping, ensuring consistent order within each group.
        
        Returns:
        - numpy.ndarray: A tensor where each matrix corresponds to one unique value of index_col, with consistent ordering.
        """
        df = self.df.copy()
        
        # Define value columns if not specified
        if value_cols is None:
            value_cols = [col for col in df.columns if col != index_col]
        
        # Sorting the DataFrame
        if sort_cols is None:
            df.sort_values(by=[index_col], inplace=True)
        else:
            df.sort_values(by=sort_cols, inplace=True)
        
        # Convert index_col to 'category' to ensure groupby treats it efficiently
        df[index_col] = df[index_col].astype('category')
        
        # Group by index_col and convert each group to a numpy array
        grouped = df.groupby(index_col)
        matrices = [group[value_cols].to_numpy() for _, group in grouped]
        
        # Find the maximum number of rows and columns for padding
        max_rows = max(matrix.shape[0] for matrix in matrices)
        
        # Pad matrices to ensure they all have the same shape
        padded_matrices = [np.pad(matrix, ((0, max_rows - matrix.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for matrix in matrices]
        
        # Stack padded matrices to form a tensor
        tensor = np.stack(padded_matrices)
        
        return tensor

    def add_bs_call_price(self):
        df = self.df.copy()
        days_in_year = 365.0
    
    # Convert inputs to the correct format
        df['T'] = df['DTE'] / days_in_year  # Time to expiration in years
        df['r'] = df['TB3MS'] / 100  # Risk-free rate as a decimal
        df['sigma'] = df['C_IV']   # Volatility as a decimal
    
    # Calculate d1 and d2
        df['d1'] = (np.log(df['UNDERLYING_LAST'] / df['STRIKE']) + (df['r'] + 0.5 * df['sigma']**2) * df['T']) / (df['sigma'] * np.sqrt(df['T']))
        df['d2'] = df['d1'] - df['sigma'] * np.sqrt(df['T'])
    
    # Calculate call option price
        df['BS_CALL_PRICE'] = df['UNDERLYING_LAST'] * norm.cdf(df['d1']) - df['STRIKE'] * np.exp(-df['r'] * df['T']) * norm.cdf(df['d2'])
    
        return df

    def black_scholes_call_vectorized(S, K, T, r, sigma):
        """Vectorized Black-Scholes call option price calculation."""
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def objective_r(r, S, K, T, sigma, C_ask):
        """Objective function to find the root (risk-free rate r)."""
        return OPML.black_scholes_call_vectorized(S, K, T, r, sigma) - C_ask

    @staticmethod
    def objective_r_brentq(r, S, K, T, sigma, C_ask):
        """Objective function for brentq method."""
        return OPML.black_scholes_call_vectorized(S, K, T, r, sigma) - C_ask

    def deduce_r_vectorized(self, r_guess=0.005, tol=1e-5, maxiter=100):
        """Deduce risk-free rate r for each row in the DataFrame."""
        S = self.df['UNDERLYING_LAST'].values
        K = self.df['STRIKE'].values
        T = self.df['DTE'].values / 365
        sigma = self.df['C_IV'].values
        C_ask = self.df['C_ASK'].values

        # Updated to use brentq for root finding and error handling
        r_deduced = []
        for i in range(len(S)):
            try:
                r = brentq(OPML.objective_r_brentq, a=-0.1, b=0.1, args=(S[i], K[i], T[i], sigma[i], C_ask[i]), xtol=tol, maxiter=maxiter)
                r_deduced.append(r)
            except ValueError as e:
                # Handle the case where brentq fails to find a root
                print(f"Root finding failed for row {i}: {e}")
                r_deduced.append(None)  # Append None or consider a default value

        self.df['deduced_r'] = r_deduced
    @staticmethod
    def objective_r(r, S, K, T, sigma, C_ask):
        """Objective function for root finding."""
        return OPML.black_scholes_call_vectorized(S, K, T, r, sigma) - C_ask

    @staticmethod
    def deduce_r_for_row(row, initial_guess=0.05):
        """Deduce risk-free rate for a single row using a fallback strategy."""
        S, K, T, sigma, C_ask = row['UNDERLYING_LAST'], row['STRIKE'], row['DTE'] / 365, row['C_IV'], row['C_ASK']
        try:
            r = brentq(OPML.objective_r, initial_guess - 0.05, initial_guess + 0.05, args=(S, K, T, sigma, C_ask))
        except ValueError:
            try:
                r = newton(OPML.objective_r, initial_guess, args=(S, K, T, sigma, C_ask))
            except RuntimeError:
                r = None  # Indicate failure or apply another fallback strategy
        return r

    def deduce_r_fallback_multithreaded(self, initial_guess=0.05):
        """Deduce risk-free rate r using a fallback strategy with multithreading."""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.deduce_r_for_row, row, initial_guess) for _, row in self.df.iterrows()]
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Deducing Risk-Free Rates"):
                results.append(future.result())
        
        self.df['deduced_r'] = results
        return self.df


    def deduce_r_fallback_multithreaded_batched(self, initial_guess=0.01, batch_size=10000):
        """Deduce risk-free rate r using a fallback strategy with multithreading, processed in batches."""
        num_rows = len(self.df)
        batches = (num_rows - 1) // batch_size + 1  # Calculate how many batches are needed
        
        # Initialize an empty list to hold the deduced risk-free rates
        deduced_r = []

        for batch in tqdm(range(batches), desc="Processing Batches"):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_rows)
            batch_df = self.df.iloc[start:end]

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.deduce_r_for_row, row, initial_guess) for _, row in batch_df.iterrows()]
                for future in as_completed(futures):
                    deduced_r.append(future.result())
                    
        # Once all batches are processed, assign the deduced rates back to the DataFrame
        self.df['deduced_r'] = deduced_r
        return self.df


