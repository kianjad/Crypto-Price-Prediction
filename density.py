import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.integrate import trapezoid


def black_scholes_call(S, K, T, r, sigma):
    """
    Price a European call option using Black-Scholes.
    S     : current price of underlying
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate (annualized)
    sigma : implied volatility (annualized), as a decimal e.g. 0.50 for 50%
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def compute_rnd(df, S, T, r=0.05, strike_min=40000, strike_max=140000):
    """
    Derive the risk-neutral density using Breeden-Litzenberger.

    Steps:
      1. Filter to OTM options only within our strike range
      2. Smooth implied volatility as a function of strike using a spline
      3. Convert smoothed implied vols -> call prices via Black-Scholes
      4. Differentiate twice -> risk-neutral density
    """
    atm   = S
    calls = df[(df['option_type'] == 'call') & (df['strike'] > atm) &
               (df['strike'] >= strike_min) & (df['strike'] <= strike_max)].copy()
    puts  = df[(df['option_type'] == 'put')  & (df['strike'] < atm) &
               (df['strike'] >= strike_min) & (df['strike'] <= strike_max)].copy()

    otm = pd.concat([puts, calls]).sort_values('strike').reset_index(drop=True)
    otm['sigma'] = otm['implied_vol'] / 100.0

    strikes = otm['strike'].values
    sigmas  = otm['sigma'].values

    vol_spline  = UnivariateSpline(strikes, sigmas, k=4, s=len(strikes) * 0.001)
    K_grid      = np.linspace(strike_min, strike_max, 500)
    sigma_grid  = np.clip(vol_spline(K_grid), 0.01, None)

    call_grid   = np.array([black_scholes_call(S, K, T, r, sig)
                             for K, sig in zip(K_grid, sigma_grid)])

    call_spline = UnivariateSpline(K_grid, call_grid, k=4, s=0)
    d2C_dK2     = call_spline.derivative(n=2)(K_grid)

    density = np.exp(r * T) * d2C_dK2
    density = np.clip(density, 0, None)
    density /= trapezoid(density, K_grid)

    return pd.DataFrame({'strike': K_grid, 'density': density})