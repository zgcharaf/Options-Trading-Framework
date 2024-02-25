import numpy as np
import pandas as pd
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ImpliedVolatilityCalculator:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def binomial_tree_american(S, K, T, r, sigma, N, option_type):
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Initialize price tree
        price_tree = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(i + 1):
                price_tree[j, i] = S * (u ** j) * (d ** (i - j))
        
        # Initialize option value tree
        option_tree = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            if option_type == 'call':
                option_tree[i, N] = max(0, price_tree[i, N] - K)
            else:  # put
                option_tree[i, N] = max(0, K - price_tree[i, N])
        
        # Backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                exercise_val = max(price_tree[j, i] - K, 0) if option_type == 'call' else max(K - price_tree[j, i], 0)
                option_tree[j, i] = max(discount * (p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1]), exercise_val)
        
        return option_tree[0, 0]
    def implied_volatility(self, S, K, T, r, market_price, sigma_guess=0.2, option_type='call'):
        def objective(sigma):
            return self.binomial_tree_american(S, K, T, r, sigma, N=100, option_type=option_type) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-20)
        except ValueError:
            iv = np.nan  # If IV cannot be found
        return iv

    def _calculate_iv_row(self, row):
        if pd.isna(row['C_IV']) :
            row['C_IV'] = self.implied_volatility(
                row['UNDERLYING_LAST'], row['STRIKE'], row['DTE'] / 365, row['r'],
                row['C_LAST'], option_type='call')
        if pd.isna(row['P_IV']):
            row['P_IV'] = self.implied_volatility(
                row['UNDERLYING_LAST'], row['STRIKE'], row['DTE'] / 365, row['r'],
                row['P_LAST'], option_type='put')
        return row

    def calculate_iv_multithreaded(self, num_threads=4):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(self._calculate_iv_row, [row for _, row in self.df.iterrows()]), total=len(self.df)))
        self.df = pd.DataFrame(results)


