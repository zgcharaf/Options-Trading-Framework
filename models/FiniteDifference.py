import numpy as np
import pandas as pd

class FiniteDifference:
    def __init__(self, df):
        """
        Initializes the OptionPricer with a DataFrame containing option and underlying asset data.
        
        :param df: DataFrame with columns for underlying asset price ('UNDERLYING_LAST'), 
                   strike price ('STRIKE'), risk-free rate ('r'), days to expiration ('DTE'), 
                   and volatility ('sigma').
        """
        self.df = df
   
        

    def finite_difference_option_price(self, S, X, r, T, sigma, N, M, option_type=None, american=True):
        """Prices an option using the finite difference method.
        
        :param S: Current price of the underlying asset.
        :param X: Exercise (strike) price of the option.
        :param r: Risk-free interest rate.
        :param T: Time to expiration of the option (in years).
        :param sigma: Volatility of the underlying asset's price.
        :param N: Number of time steps in the finite difference grid.
        :param M: Number of asset price steps in the finite difference grid.
        :param option_type: Type of the option ('call' or 'put').
        :param american: Boolean indicating if the option is American (True) or European (False).
        :return: The priced value of the option.
        """
        dt = T/N
        dx = sigma * np.sqrt(dt)
        pu = 0.5 * dt * ((sigma**2 / dx**2) + (r - 0.5 * sigma**2) / dx)
        pm = 1 - dt * (sigma**2 / dx**2 + r)
        pd = 0.5 * dt * ((sigma**2 / dx**2) - (r - 0.5 * sigma**2) / dx)
        grid = np.zeros((M+1, N+1))
        stock_prices = S * np.exp(np.arange(-M/2, M/2 + 1) * dx)
        
        # Terminal conditions
        if option_type == 'call':
            grid[:, -1] = np.maximum(stock_prices - X, 0)
        elif option_type == 'put':
            grid[:, -1] = np.maximum(X - stock_prices, 0)

        # Iterate backwards in time
        for i in range(N-1, -1, -1):
            for j in range(1, M):
                grid[j, i] = pu * grid[j+1, i+1] + pm * grid[j, i+1] + pd * grid[j-1, i+1]
                if american:
                    # Check for early exercise
                    if option_type == 'call':
                        grid[j, i] = max(grid[j, i], stock_prices[j] - X)
                    elif option_type == 'put':
                        grid[j, i] = max(grid[j, i], X - stock_prices[j])

        # Return the option price at S
        return grid[M//2, 0]

    def price_options(self, N=50, M=50, american=True, option_type='call'):
        """
        Applies the finite difference option pricing method to each row in the DataFrame.
        
        :param N: Number of time steps.
        :param M: Number of price steps.
        :param american: Boolean indicating if options are American.
        :param option_type: Specifies the option type to price ('call' or 'put').
        """
        # Apply the pricing method to each row in the DataFrame
        self.df['Option_Price_FD'] = self.df.apply(
            lambda row: self.finite_difference_option_price(
                S=row['UNDERLYING_LAST'],
                X=row['STRIKE'],
                r=row['r'],
                T=row['DTE'] / 365,  # Convert DTE to years
                sigma=row['sigma'],
                N=N,
                M=M,
                option_type=option_type,  # Assumes 'call' or 'put' is specified in the DataFrame
                american=american  # Whether to consider it as an American option for early exercise
            ),
            axis=1
        )
"""
# Generate a synthetic dataset
np.random.seed(42)  # For reproducibility
data = {
    'UNDERLYING_LAST': np.random.uniform(90, 110, 10),  # Random stock prices between 90 and 110
    'STRIKE': np.random.uniform(95, 105, 10),  # Random strike prices between 95 and 105
    'r': np.random.uniform(0.01, 0.05, 10),  # Random risk-free rates between 1% and 5%
    'DTE': np.random.randint(30, 365, 10),  # Random days to expiration between 30 and 365
    'sigma': np.random.uniform(0.15, 0.3, 10),  # Random volatilities between 15% and 30%
    'option_type': ['call'] * 5 + ['put'] * 5  # 5 calls and 5 puts
}
df = pd.DataFrame(data)

# Instantiate the OptionPricer with the synthetic dataset
pricer = OptionPricer(df)

# Price the options
pricer.price_options(N=50, M=50, american=True, option_type='call')  # For calls
pricer.price_options(N=50, M=50, american=True, option_type='put')  # For puts

# Display the DataFrame with the calculated option prices
print(df[['UNDERLYING_LAST', 'STRIKE', 'r', 'DTE', 'sigma', 'option_type', 'Option_Price_FD']])"""
