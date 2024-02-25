import numpy as np 
import pandas as pd 

class BinomialTree:
    def __init__(self, df):
        """
        Initializes the OptionPricer with a DataFrame containing option and underlying asset data.
        """
        self.df = df

    def binomial_tree_option_price(self, S, X, r, T, sigma, N, option_type='call', american=True):
        """
        Prices an option using the binomial tree model.
        
        Parameters:
        - S: Current price of the underlying asset.
        - X: Exercise (strike) price of the option.
        - r: Risk-free interest rate.
        - T: Time to expiration of the option (in years).
        - sigma: Volatility of the underlying asset's price.
        - N: Number of steps in the binomial tree.
        - option_type: Type of the option ('call' or 'put').
        - american: Boolean indicating if the option is American (True) or European (False).
        
        Returns:
        - The priced value of the option.
        """
        dt = T / N  # Time step
        u = np.exp(sigma * np.sqrt(dt))  # Upward movement factor
        d = 1 / u  # Downward movement factor
        p = (np.exp(r * dt) - d) / (u - d)  # Probability of upward movement
        discount = np.exp(-r * dt)  # Discount factor per time step

        # Initialize asset prices at maturity
        asset_prices = np.asarray([S * u**j * d**(N - j) for j in range(N + 1)])
        
        # Initialize option values at maturity
        if option_type == 'call':
            option_values = np.maximum(asset_prices - X, 0)
        else:  # put
            option_values = np.maximum(X - asset_prices, 0)
        
        # Iterate backwards through the tree
        for i in range(N - 1, -1, -1):
            option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
            if american:
                # Check for early exercise
                asset_prices = S * u**i * d**(N - i)
                if option_type == 'call':
                    option_values = np.maximum(option_values, asset_prices - X)
                else:
                    option_values = np.maximum(option_values, X - asset_prices)
        
        return option_values[0]  # Return the option price at the root of the tree

    def price_options_binomial(self, N=50, option_type='call', american=True):
        """
        Applies the binomial tree option pricing method to each row in the DataFrame.
        """
        self.df['Option_Price_BT'] = self.df.apply(
            lambda row: self.binomial_tree_option_price(
                S=row['UNDERLYING_LAST'],
                X=row['STRIKE'],
                r=row['r'],
                T=row['DTE'] / 365,
                sigma=row['sigma'],
                N=N,
                option_type=option_type,
                american=american),
            axis=1
        )
