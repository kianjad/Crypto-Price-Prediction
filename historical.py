import requests
import numpy as np
import pandas as pd

def get_historical_prices(coin='bitcoin', days=365):
    """
    Pull daily closing prices from CoinGecko.
    coin: 'bitcoin' or 'ethereum'
    days: number of days of history (730 = 2 years)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'prices' not in data:
        raise ValueError(f"CoinGecko error: {data}")    

    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms').dt.date
    prices = prices[['date', 'price']].set_index('date')
    return prices

def compute_log_return_params(prices):
    """
    Estimate mu and sigma of daily log-returns.
    Under lognormal assumption, log(S_T/S_0) ~ N(mu*T, sigma^2*T)
    where mu and sigma are annualized.
    """
    log_returns = np.log(prices['price'] / prices['price'].shift(1)).dropna()

    mu_daily    = log_returns.mean()
    sigma_daily = log_returns.std()

    # Annualize
    mu_annual    = mu_daily * 365
    sigma_annual = sigma_daily * np.sqrt(365)

    return mu_annual, sigma_annual, log_returns

def get_realworld_density(S, T, mu, sigma, strike_min=40000, strike_max=140000, n=500):
    """
    Compute the lognormal real-world density of BTC price at time T.
    S     : current price
    T     : time to expiry in years
    mu    : annualized log-return mean
    sigma : annualized log-return std
    """
    from scipy.stats import norm

    K_grid = np.linspace(strike_min, strike_max, n)

    # Log-price at T is normally distributed
    # log(S_T) ~ N(log(S) + mu*T, sigma^2*T)
    mu_logprice    = np.log(S) + mu * T
    sigma_logprice = sigma * np.sqrt(T)

    # Lognormal density: f(K) = (1 / K*sigma_logprice) * phi((log(K) - mu_logprice) / sigma_logprice)
    density = norm.pdf((np.log(K_grid) - mu_logprice) / sigma_logprice) / (K_grid * sigma_logprice)

    # Normalize
    density /= np.trapz(density, K_grid)

    return pd.DataFrame({'strike': K_grid, 'density': density})

if __name__ == "__main__":
    prices = get_historical_prices('bitcoin', days=365)
    mu, sigma, log_returns = compute_log_return_params(prices)

    print(f"BTC annualized log-return mean:  {mu:.4f} ({mu*100:.1f}%)")
    print(f"BTC annualized log-return sigma: {sigma:.4f} ({sigma*100:.1f}%)")
    print(f"Based on {len(log_returns)} daily observations")