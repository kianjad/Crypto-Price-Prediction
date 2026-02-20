import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date
from data import get_option_chain
from density import compute_rnd
from historical import get_historical_prices, compute_log_return_params, get_realworld_density

def get_spot_price(currency='ETH'):
    url = "https://www.deribit.com/api/v2/public/get_index_price"
    resp = requests.get(url, params={"index_name": f"{currency.lower()}_usd"}).json()
    return resp['result']['index_price']

S = get_spot_price('ETH')

T = 36 / 365
r = 0.05
strike_min = 800
strike_max = 4000


# Deribit uses ETH- prefix for ETH options
# We need to pull ETH option chain specifically
import requests

def get_eth_option_chain(target_date_str):
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    data = requests.get(url, params={"currency": "ETH", "kind": "option"}).json()
    instruments = data['result']

    target_date = pd.Timestamp(target_date_str).date()
    filtered = [
        i for i in instruments
        if pd.to_datetime(i['expiration_timestamp'], unit='ms').date() == target_date
    ]
    print(f"Found {len(filtered)} ETH options for {target_date_str}")

    rows = []
    for inst in filtered:
        resp = requests.get(
            "https://www.deribit.com/api/v2/public/ticker",
            params={"instrument_name": inst['instrument_name']}
        ).json()
        ticker = resp['result']
        rows.append({
            'instrument': inst['instrument_name'],
            'strike':      inst['strike'],
            'option_type': inst['option_type'],
            'implied_vol': ticker.get('mark_iv', None),
            'mark_price':  ticker.get('mark_price', None),
            'bid':         ticker.get('best_bid_price', None),
            'ask':         ticker.get('best_ask_price', None),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['implied_vol'])
    df = df[df['implied_vol'] > 0]
    df = df.sort_values('strike').reset_index(drop=True)
    return df

df  = get_eth_option_chain('2026-03-27')
rnd = compute_rnd(df, S=S, T=T, r=r, strike_min=strike_min, strike_max=strike_max)

# Real-world density

prices       = get_historical_prices('ethereum', days=365)
mu, sigma, _ = compute_log_return_params(prices)
rwd          = get_realworld_density(S, T, mu, sigma,
                                     strike_min=strike_min, strike_max=strike_max)

print(f"ETH annualized mu: {mu*100:.1f}%  sigma: {sigma*100:.1f}%")

# Summary statistics

def density_stats(strikes, density):
    mean = np.trapz(strikes * density, strikes)
    var  = np.trapz((strikes - mean)**2 * density, strikes)
    std  = np.sqrt(var)
    skew = np.trapz(((strikes - mean)/std)**3 * density, strikes)
    kurt = np.trapz(((strikes - mean)/std)**4 * density, strikes) - 3
    cdf  = np.array([np.trapz(density[:i+1], strikes[:i+1]) for i in range(len(strikes))])
    p5   = strikes[np.searchsorted(cdf, 0.05)]
    p95  = strikes[np.searchsorted(cdf, 0.95)]
    return dict(mean=mean, std=std, skew=skew, kurt=kurt, p5=p5, p95=p95)

rnd_stats = density_stats(rnd['strike'].values, rnd['density'].values)
rwd_stats = density_stats(rwd['strike'].values, rwd['density'].values)

print("=== Risk-Neutral (Q) ===")
for k, v in rnd_stats.items():
    print(f"  {k}: {v:,.2f}")
print("=== Real-World (P) ===")
for k, v in rwd_stats.items():
    print(f"  {k}: {v:,.2f}")

# Plot

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=rnd['strike'], y=rnd['density'],
    mode='lines',
    line=dict(color='#1a47a8', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(26, 71, 168, 0.12)',
    name='Risk-Neutral (Q-measure)'
))

fig.add_trace(go.Scatter(
    x=rwd['strike'], y=rwd['density'],
    mode='lines',
    line=dict(color='#c0392b', width=2.5, dash='dash'),
    fill='tozeroy',
    fillcolor='rgba(192, 57, 43, 0.08)',
    name='Real-World (P-measure)'
))

fig.add_vline(x=S, line_dash='dot', line_color='grey', line_width=1.5,
              annotation_text=f'Current Price ${S:,}',
              annotation_position='top right',
              annotation_font_size=11)

fig.update_layout(
    title=dict(
        text='Ethereum: Risk-Neutral vs Real-World Probability Density<br>'
             '<sup>Q-measure from Breeden-Litzenberger · P-measure from historical log-returns · Mar 27 2026 Expiry</sup>',
        font=dict(size=16),
        x=0.05
    ),
    xaxis=dict(
        title='ETH Price at Expiry (USD)',
        tickprefix='$', tickformat=',',
        gridcolor='#eeeeee', showline=True, linecolor='#cccccc'
    ),
    yaxis=dict(
        title='Probability Density',
        gridcolor='#eeeeee', showline=True, linecolor='#cccccc'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(x=0.98, y=0.97, xanchor='right',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='#cccccc', borderwidth=1),
    hovermode='x unified',
    width=1100, height=600,
    margin=dict(l=60, r=40, t=100, b=60)
)

fig.show()