import plotly.graph_objects as go
import numpy as np
import requests
from data import get_option_chain
from density import compute_rnd

def get_spot_price(currency='BTC'):
    url = "https://www.deribit.com/api/v2/public/get_index_price"
    resp = requests.get(url, params={"index_name": f"{currency.lower()}_usd"}).json()
    return resp['result']['index_price']

S = get_spot_price('BTC')
T = 36 / 365
r = 0.05

df = get_option_chain('2026-03-27')
rnd = compute_rnd(df, S=S, T=T, r=r)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=rnd['strike'],
    y=rnd['density'],
    mode='lines',
    line=dict(color='royalblue', width=2),
    fill='tozeroy',
    fillcolor='rgba(65, 105, 225, 0.15)',
    name='Risk-Neutral Density'
))

fig.add_vline(
    x=S,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Current Price ${S:,}',
    annotation_position='top right'
)

fig.update_layout(
    title='Bitcoin Risk-Neutral Probability Density (Mar 27 2026 Expiry)',
    xaxis_title='BTC Price at Expiry ($)',
    yaxis_title='Probability Density',
    template='plotly_white',
    hovermode='x unified'
)

fig.show()