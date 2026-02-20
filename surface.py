import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date
from data import get_option_chain, get_available_expiries
from density import compute_rnd, black_scholes_call

# Parameters

def get_spot_price(currency='BTC'):
    url = "https://www.deribit.com/api/v2/public/get_index_price"
    resp = requests.get(url, params={"index_name": f"{currency.lower()}_usd"}).json()
    return resp['result']['index_price']

S = get_spot_price('BTC')
r = 0.05
today = date(2026, 2, 19)

strike_min = 40000
strike_max = 140000

# Loop over all expiries

expiries = get_available_expiries()
print(f"Found {len(expiries)} expiries")

results = []

for expiry in expiries:
    T = (expiry - today).days / 365
    if T <= 35/365:
        print(f"Skipping {expiry} — expires too soon")
        continue

    print(f"Processing {expiry} (T = {T:.3f} years)...")
    try:
        df = get_option_chain(str(expiry))
        rnd = compute_rnd(df, S=S, T=T, r=r,
                          strike_min=strike_min, strike_max=strike_max)
        rnd['expiry'] = str(expiry)
        rnd['days_to_expiry'] = (expiry - today).days
        results.append(rnd)
    except Exception as e:
        print(f"  Failed: {e}")
        continue

print(f"\nSuccessfully computed {len(results)} densities")

# Build 3D surface

# Stack into a matrix: rows = expiries, cols = strike grid
expiry_labels = [r['expiry'].iloc[0] for r in results]
days_list     = [r['days_to_expiry'].iloc[0] for r in results]
strike_grid   = results[0]['strike'].values  # same grid for all expiries

# Build density matrix (n_expiries x n_strikes)
Z = np.array([r['density'].values for r in results])

# Interpolate between expiries to make hover tracking smooth
from scipy.interpolate import interp1d

days_array = np.array(days_list)
days_fine  = np.linspace(days_array.min(), days_array.max(), 60)

Z_interp = np.zeros((len(days_fine), len(strike_grid)))
for i, k in enumerate(range(len(strike_grid))):
    f = interp1d(days_array, Z[:, k], kind='cubic')
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

# Mark current spot price
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
        text='Bitcoin Risk-Neutral Probability Density Surface<br>'
             '<sup>Breeden-Litzenberger · Deribit Options · All Expiries</sup>',
        font=dict(size=16),
        x=0.05
    ),
    scene=dict(
    xaxis=dict(
        title=dict(text='<br><br>BTC Price at Expiry ($)', font=dict(size=13)),
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
    camera=dict(
        eye=dict(x=1.8, y=-1.8, z=0.8)
    ),
    aspectratio=dict(x=1.5, y=1.2, z=0.8),
    ),
    width=1200,
    height=700,
    autosize=False,
    paper_bgcolor='white',
    legend=dict(x=0.8, y=0.9),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.show()