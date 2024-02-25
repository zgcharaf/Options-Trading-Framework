import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize, fftpack
from scipy.special import gamma

class CGMY:
    def __init__(self, r=0.05, alpha=1.1, Sigma=0.25):
        self.r = r
        self.alpha = alpha
        self.Sigma = Sigma
        self.T = None
        self.DiscountFactor = None
        self.FFT_N = None
        self.FFT_eta = None
        self.FFT_lambda = None
        self.FFT_b = None
        self.uvec = None
        self.ku = None
        self.vj = None
        self.jvec = None
        self.GLOBALFAKTOR = None
        self.GLOBALKU = None
        self.GLOBALCPVEC = None

    def cf_log_cgmy(self, u, lnS, T, mu, half_etasq, C, G, M, Y):
        omega = -C * gamma(-Y) * (np.power(M-1, Y) - np.power(M, Y) + np.power(G+1, Y) - np.power(G, Y))
        phi_CGMY = C * T * gamma(-Y) * (np.power(M-1j*u, Y) - np.power(M, Y) + np.power(G+1j*u, Y) - np.power(G, Y))
        phi = 1j*u*(lnS + (mu+omega-half_etasq)*T) + phi_CGMY - half_etasq*np.power(u, 2)
        return np.exp(self.Sigma*phi)

    def call_price_CM_precalc(self, T, N=15):
        self.T = T
        self.DiscountFactor = np.exp(-self.r * T)
        self.FFT_N = int(np.power(2, N))
        self.FFT_eta = .05
        self.FFT_lambda = (2 * np.pi) / (self.FFT_N * self.FFT_eta)
        self.FFT_b = (self.FFT_N * self.FFT_lambda) / 2
        self.uvec = np.linspace(1, self.FFT_N, self.FFT_N)
        self.ku = -self.FFT_b + self.FFT_lambda * (self.uvec - 1)
        self.jvec = self.uvec
        self.vj = (self.uvec - 1) * self.FFT_eta
        self.GLOBALFAKTOR = self.DiscountFactor * np.exp(1j * self.vj * self.FFT_b) * self.FFT_eta
        self.GLOBALKU = self.ku

    def call_price_CM_CF(self, CF, lnS):
        tmp = self.GLOBALFAKTOR * self.psi(CF, self.vj, self.alpha, lnS, self.T)
        tmp = (tmp / 3) * (3 + np.power(-1, self.jvec) - ((self.jvec - 1) == 0))
        self.GLOBALCPVEC = np.real(np.exp(-self.alpha * self.GLOBALKU) * fftpack.fft(tmp) / np.pi)

    def call_price_CF_K(self, lnK):
        if np.isscalar(lnK):
            lnK = np.array([lnK])  # Convert scalar to an array for uniform processing
        indexOfStrike = np.floor((lnK + self.FFT_b) / self.FFT_lambda + 1).astype(int)
        prices = np.zeros_like(lnK)
        for i, idx in enumerate(indexOfStrike):
            if idx >= len(self.GLOBALKU) - 1:  # Check bounds
                prices[i] = np.nan  # or handle the edge case appropriately
            else:
                xp = [self.GLOBALKU[idx], self.GLOBALKU[idx + 1]]
                yp = [self.GLOBALCPVEC[idx], self.GLOBALCPVEC[idx + 1]]
                prices[i] = np.interp(lnK[i], xp, yp)
        return prices if len(prices) > 1 else prices[0]  # Return a scalar if input was scalar


#    def psi(self, CF, vj, alpha, lnS, T):
 #       u = vj - (alpha * 1j + 1j)
  #      denom = alpha**2 + alpha - self.Sigma**2 + vj * 2 * alpha * 1j + 1j * vj
   #     return CF(u, lnS, T, self.r, 0, 24.79, 94.45, 95.79, 0.2495) / denom

    def psi(self, CF, vj, alpha, lnS, T):
        u = vj - (alpha * 1j + 1j)
        denom = alpha**2 + alpha - self.Sigma**2 + vj * 2 * alpha * 1j + 1j * vj
    # Call CF with only the parameters it expects
        return CF(u, lnS, T) / denom

    def do_CM_CGMY_fit(self, odate, options_df):
        """
        Fit CGMY model parameters to market data from a DataFrame, using options data
        filtered by the quotation date, and stock price from the 'UNDERLYING_LAST' column.

        :param odate: The observation date as a string, e.g., '2018-08-08'.
        :param options_df: DataFrame containing options data.
        """
        Tmt = 1 / 12
        self.call_price_CM_precalc(T=Tmt, N=17)
        
        # Filter options_df for the specific observation date
        filtered_options_df = options_df[options_df['QUOTE_DATE'] == odate]
        
        if filtered_options_df.empty:
            raise ValueError(f"No data found for the specified date: {odate}")
        
        # Assuming 'UNDERLYING_LAST' contains the stock price for the observation date
        # Here we take the first value assuming all entries for the date have the same underlying price
        stock_price = filtered_options_df['UNDERLYING_LAST'].iloc[0]
        lnS = np.log(stock_price)
        
        prices_oom = pd.to_numeric(filtered_options_df['C_LAST'].values, errors='coerce')
        strike_oom = pd.to_numeric(filtered_options_df['STRIKE'].values, errors='coerce')

        
        def minimize(param):
    # Define a wrapper for cf_log_cgmy to be used with the optimization function
            def CF(u, lnS, T):
        # Unpack the parameters correctly within the cf_log_cgmy call
                return self.cf_log_cgmy(u, lnS, T, self.r, param[4], param[0], param[1], param[2], param[3])

    # Call the CM pricing function with the CF defined above
            self.call_price_CM_CF(CF, lnS)
    # Calculate the objective function value (sum of squared errors)
            return np.sum((self.call_price_CF_K(np.log(strike_oom)) - prices_oom) ** 2)
    

        
        bounds = ((1e-3, np.inf), (1e-3, np.inf), (1e-3, np.inf), (-np.inf, 2 - 1e-3), (1e-3, 1 - 1e-3))
        param = [24.79, 94.45, 95.79, 0.2495, 0]
        
        local_opt = sp.optimize.minimize(minimize, param, bounds=bounds)
        global_opt = sp.optimize.basinhopping(minimize, param, minimizer_kwargs={'bounds': bounds})
        
        return {'opt': global_opt, 'local_opt': local_opt}



