import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date
from density import compute_rnd, black_scholes_call
from scipy.interpolate import interp1d
import requests

# Parameters

def get_spot_price(currency='ETH'):
    url = "https://www.deribit.com/api/v2/public/get_index_price"
    resp = requests.get(url, params={"index_name": f"{currency.lower()}_usd"}).json()
    return resp['result']['index_price']

S = get_spot_price('ETH')
r          = 0.05
today      = date(2026, 2, 19)
strike_min = 800
strike_max = 4000

# Pull ETH option chain

def get_eth_instruments():
    url  = "https://www.deribit.com/api/v2/public/get_instruments"
    data = requests.get(url, params={"currency": "ETH", "kind": "option"}).json()
    return data['result']

def get_eth_expiries():
    instruments = get_eth_instruments()
    expiries    = sorted(set(i['expiration_timestamp'] for i in instruments))
    return pd.to_datetime(expiries, unit='ms').date

def get_eth_option_chain(target_date_str):
    instruments = get_eth_instruments()
    target_date = pd.Timestamp(target_date_str).date()
    filtered    = [
        i for i in instruments
        if pd.to_datetime(i['expiration_timestamp'], unit='ms').date() == target_date
    ]
    print(f"Found {len(filtered)} ETH options for {target_date_str}")
    rows = []
    for inst in filtered:
        resp   = requests.get(
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

# Loop over expiries

expiries = get_eth_expiries()
print(f"Found {len(expiries)} expiries")

results = []

for expiry in expiries:
    T = (expiry - today).days / 365
    if T <= 35/365:
        print(f"Skipping {expiry} — expires too soon")
        continue
    print(f"Processing {expiry} (T = {T:.3f} years)...")
    try:
        df  = get_eth_option_chain(str(expiry))
        rnd = compute_rnd(df, S=S, T=T, r=r,
                          strike_min=strike_min, strike_max=strike_max)
        rnd['expiry']         = str(expiry)
        rnd['days_to_expiry'] = (expiry - today).days
        results.append(rnd)
    except Exception as e:
        print(f"  Failed: {e}")
        continue

print(f"\nSuccessfully computed {len(results)} densities")

# Build 3D surface

days_list   = [r['days_to_expiry'].iloc[0] for r in results]
strike_grid = results[0]['strike'].values
Z           = np.array([r['density'].values for r in results])

days_array  = np.array(days_list)
days_fine   = np.linspace(days_array.min(), days_array.max(), 60)

Z_interp = np.zeros((len(days_fine), len(strike_grid)))
for i in range(len(strike_grid)):
    f            = interp1d(days_array, Z[:, i], kind='cubic')
    Z_interp[:, i] = f(days_fine)
Z_interp = np.clip(Z_interp, 0, None)

fig = go.Figure(data=[go.Surface(
    x=strike_grid,
    y=days_fine,
    z=Z_interp,
    name='RND Surface',
    colorscale='Viridis',
    reversescale=False,
    opacity=0.9,
    contours=dict(
        z=dict(show=True, usecolormap=True, highlightcolor='white', project_z=True)
    )
)])

fig.add_scatter3d(
    x=[S, S],
    y=[min(days_list), max(days_list)],
    z=[0, 0],
    mode='lines',
    line=dict(color='red', width=4),
    name=f'Current Price ${S:,}'
)

fig.update_layout(
    title=dict(
        text='Ethereum Risk-Neutral Probability Density Surface<br>'
             '<sup>Breeden-Litzenberger · Deribit Options · All Expiries</sup>',
        font=dict(size=16),
        x=0.05
    ),
    scene=dict(
        xaxis=dict(
            title=dict(text='<br><br>ETH Price at Expiry ($)', font=dict(size=13)),
            tickformat='$,.0f',
            gridcolor='lightgrey',
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(text='<br>Days to Expiry'),
            gridcolor='lightgrey',
            autorange='reversed',
        ),
        zaxis=dict(
            title=dict(text='Probability Density'),
            gridcolor='lightgrey',
        ),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
        aspectratio=dict(x=1.5, y=1.2, z=0.8),
    ),
    width=1000,
    height=620,
    autosize=False,
    paper_bgcolor='white',
    legend=dict(x=0.8, y=0.9),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.show()