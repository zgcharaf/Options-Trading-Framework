# __init__.py in the root of options_pricing_library

# Package-wide imports
from .models.black_scholes import BlackScholes
from .models.binomial_tree import BinomialTree
from .models.monte_carlo import MonteCarlo
from .models.finite_difference import FiniteDifference
from .api.pricing_api import PricingAPI
from .utils.utilss import (
    current_time_formatted,
    days_from_now,
    read_data_csv,
    clean_and_transform_market_data,
    get_from_fred_with_fredapi,
    process_and_merge_data
)

# Version of the options_pricing_library package
__version__ = '1.0.0'

# Optionally, initialize logging or any other package-wide setups here

# Convenience imports for easier access to common functionalities
__all__ = ['BlackScholes',
           'BinomialTree', 
           'MonteCarlo', 
           'FiniteDifference', 
           'PricingAPI',
            'current_time_formatted',
            'days_from_now',
            'read_data_csv',
            'clean_and_transform_market_data',
            'get_from_fred_with_fredapi',
            'process_and_merge_data'
]
