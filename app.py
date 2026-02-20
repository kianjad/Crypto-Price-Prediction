import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import norm

st.set_page_config(
    page_title="Crypto Risk-Neutral Density",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #eff2ec; color: #1a1a1a; }

[data-testid="collapsedControl"] { display: none; }

h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 28px !important;
    color: #1a1a1a;
    letter-spacing: -0.02em;
}

.block-container { padding-top: 2.5rem; max-width: 860px; }

.stButton > button {
    background: #1a1a1a;
    color: #f5f5f0;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    padding: 10px 28px;
}

.stButton > button:hover { background: #333; }

.footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid #dde0d8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #aaa;
}

.mu-warning {
    background: #fffbe6;
    border: 1px solid #e6d87a;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 12px;
    color: #7a6a00;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# Core functions

def get_spot_price(currency='BTC'):
    url  = "https://www.deribit.com/api/v2/public/get_index_price"
    resp = requests.get(url, params={"index_name": f"{currency.lower()}_usd"}, timeout=10).json()
    return resp['result']['index_price']

def get_instruments(currency='BTC'):
    url  = "https://www.deribit.com/api/v2/public/get_instruments"
    data = requests.get(url, params={"currency": currency, "kind": "option"}, timeout=10).json()
    return data['result']

def get_available_expiries(currency='BTC'):
    instruments = get_instruments(currency)
    expiries    = sorted(set(i['expiration_timestamp'] for i in instruments))
    return pd.to_datetime(expiries, unit='ms').date

def get_option_chain(target_date_str, currency='BTC'):
    instruments = get_instruments(currency)
    target_date = pd.Timestamp(target_date_str).date()
    filtered    = [i for i in instruments
                   if pd.to_datetime(i['expiration_timestamp'], unit='ms').date() == target_date]
    rows = []
    for inst in filtered:
        resp   = requests.get("https://www.deribit.com/api/v2/public/ticker",
                              params={"instrument_name": inst['instrument_name']}, timeout=10).json()
        ticker = resp['result']
        rows.append({
            'strike':      inst['strike'],
            'option_type': inst['option_type'],
            'implied_vol': ticker.get('mark_iv', None),
        })
    df = pd.DataFrame(rows).dropna(subset=['implied_vol'])
    df = df[df['implied_vol'] > 0].sort_values('strike').reset_index(drop=True)
    return df

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def compute_rnd(df, S, T, r=0.05, strike_min=40000, strike_max=140000):
    calls = df[(df['option_type'] == 'call') & (df['strike'] > S) &
               (df['strike'] >= strike_min) & (df['strike'] <= strike_max)].copy()
    puts  = df[(df['option_type'] == 'put')  & (df['strike'] < S) &
               (df['strike'] >= strike_min) & (df['strike'] <= strike_max)].copy()
    otm   = pd.concat([puts, calls]).sort_values('strike').reset_index(drop=True)
    otm['sigma'] = otm['implied_vol'] / 100.0
    strikes     = otm['strike'].values
    sigmas      = otm['sigma'].values
    vol_spline  = UnivariateSpline(strikes, sigmas, k=4, s=len(strikes) * 0.001)
    K_grid      = np.linspace(strike_min, strike_max, 500)
    sigma_grid  = np.clip(vol_spline(K_grid), 0.01, None)
    call_grid   = np.array([black_scholes_call(S, K, T, r, sig) for K, sig in zip(K_grid, sigma_grid)])
    call_spline = UnivariateSpline(K_grid, call_grid, k=4, s=0)
    density     = np.exp(r * T) * call_spline.derivative(n=2)(K_grid)
    density     = np.clip(density, 0, None)
    density    /= trapezoid(density, K_grid)
    return pd.DataFrame({'strike': K_grid, 'density': density})

def get_historical_prices(coin='bitcoin', days=365):
    url  = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    resp = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "daily"},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    data = resp.json()
    if 'prices' not in data:
        raise ValueError(f"CoinGecko error: {data}")
    prices         = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms').dt.date
    return prices[['date', 'price']].set_index('date')

def compute_log_return_params(prices):
    log_returns = np.log(prices['price'] / prices['price'].shift(1)).dropna()
    return log_returns.mean() * 365, log_returns.std() * np.sqrt(365), log_returns

def get_realworld_density(S, T, mu, sigma, strike_min=40000, strike_max=140000):
    K_grid    = np.linspace(strike_min, strike_max, 500)
    mu_log    = np.log(S) + mu * T
    sigma_log = sigma * np.sqrt(T)
    density   = norm.pdf((np.log(K_grid) - mu_log) / sigma_log) / (K_grid * sigma_log)
    density  /= trapezoid(density, K_grid)
    return pd.DataFrame({'strike': K_grid, 'density': density})

def density_stats(strikes, density):
    mean = trapezoid(strikes * density, strikes)
    var  = trapezoid((strikes - mean)**2 * density, strikes)
    std  = np.sqrt(var)
    skew = trapezoid(((strikes - mean)/std)**3 * density, strikes)
    kurt = trapezoid(((strikes - mean)/std)**4 * density, strikes) - 3
    cdf  = np.array([trapezoid(density[:i+1], strikes[:i+1]) for i in range(len(strikes))])
    p5   = strikes[np.searchsorted(cdf, 0.05)]
    p95  = strikes[np.searchsorted(cdf, 0.95)]
    return dict(mean=mean, std=std, skew=skew, kurt=kurt, p5=p5, p95=p95)

PLOT_BGCOLOR  = '#f4f6f1'
PAPER_BGCOLOR = '#eff2ec'
GRID_COLOR    = '#e0e5db'
TICK_COLOR    = '#999'
r             = 0.05

@st.cache_data(ttl=60)
def cached_spot(asset):
    return get_spot_price(asset)

@st.cache_data(ttl=300)
def cached_expiries(asset):
    return get_available_expiries(asset)

# Header

st.title("Crypto Risk-Neutral Density")
st.markdown("Extracting market-implied price distributions from live options data · Live data")
st.markdown("---")

# Controls row

col1, col2 = st.columns([1, 2])

with col1:
    asset = st.radio("Asset", ["BTC", "ETH"], horizontal=True)

with col2:
    plot_type = st.radio("Plot type", ["Q vs P Comparison", "3D Density Surface"], horizontal=True)

coin_name  = "bitcoin" if asset == "BTC" else "ethereum"
strike_min = 40000  if asset == "BTC" else 800
strike_max = 140000 if asset == "BTC" else 4000

# Spot price

try:
    S = cached_spot(asset)
except Exception as e:
    st.error(f"Could not fetch current price: {e}")
    st.stop()

st.caption(f"{asset} current price: **${S:,.2f}**")
st.markdown("")

# Q vs P Comparison

if plot_type == "Q vs P Comparison":

    try:
        expiries     = cached_expiries(asset)
        today        = date.today()
        future_exp   = [str(e) for e in expiries if (e - today).days > 7]
        col_a, col_b = st.columns([2, 3])
        with col_a:
            selected_exp = st.selectbox("Expiry date", future_exp)
        T = (pd.Timestamp(selected_exp).date() - today).days / 365
    except Exception as e:
        st.error(f"Could not fetch expiries: {e}")
        st.stop()

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate = st.button("Generate")

    if generate:
        with st.spinner("Fetching option chain..."):
            df = get_option_chain(selected_exp, currency=asset)

        with st.spinner("Computing densities..."):
            rnd          = compute_rnd(df, S=S, T=T, r=r, strike_min=strike_min, strike_max=strike_max)
            prices       = get_historical_prices(coin_name)
            mu, sigma, _ = compute_log_return_params(prices)
            rwd          = get_realworld_density(S, T, mu, sigma,
                                                 strike_min=strike_min, strike_max=strike_max)

        rnd_stats = density_stats(rnd['strike'].values, rnd['density'].values)
        rwd_stats = density_stats(rwd['strike'].values, rwd['density'].values)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rnd['strike'], y=rnd['density'], mode='lines',
            line=dict(color='#2c5fa8', width=2.5),
            fill='tozeroy', fillcolor='rgba(44,95,168,0.10)',
            name='Risk-Neutral (Q)'
        ))
        fig.add_trace(go.Scatter(
            x=rwd['strike'], y=rwd['density'], mode='lines',
            line=dict(color='#b03030', width=2.5, dash='dash'),
            fill='tozeroy', fillcolor='rgba(176,48,48,0.07)',
            name='Real-World (P)'
        ))
        fig.add_vline(x=S, line_dash='dot', line_color='#aaa', line_width=1.5,
                      annotation_text=f'Current Price ${S:,.0f}',
                      annotation_position='top right',
                      annotation_font=dict(color='#888', size=11))
        fig.update_layout(
            title=f'{asset}: Risk-Neutral vs Real-World Density · {selected_exp} Expiry',
            xaxis=dict(title=f'{asset} Price at Expiry (USD)', tickprefix='$', tickformat=',',
                       gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR)),
            yaxis=dict(title='Probability Density', gridcolor=GRID_COLOR,
                       tickfont=dict(color=TICK_COLOR)),
            plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
            legend=dict(x=0.98, y=0.97, xanchor='right',
                        bgcolor='rgba(244,246,241,0.9)', bordercolor='#ddd', borderwidth=1),
            hovermode='x unified', height=500,
            margin=dict(l=60, r=40, t=60, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Distribution statistics**")
        stats_df = pd.DataFrame({
            'Risk-Neutral (Q)': {
                'Mean':            f"${rnd_stats['mean']:,.0f}",
                'Std Dev':         f"${rnd_stats['std']:,.0f}",
                'Skewness':        f"{rnd_stats['skew']:.3f}",
                'Excess Kurtosis': f"{rnd_stats['kurt']:.3f}",
                '5th Percentile':  f"${rnd_stats['p5']:,.0f}",
                '95th Percentile': f"${rnd_stats['p95']:,.0f}",
            },
            'Real-World (P)': {
                'Mean':            f"${rwd_stats['mean']:,.0f}",
                'Std Dev':         f"${rwd_stats['std']:,.0f}",
                'Skewness':        f"{rwd_stats['skew']:.3f}",
                'Excess Kurtosis': f"{rwd_stats['kurt']:.3f}",
                '5th Percentile':  f"${rwd_stats['p5']:,.0f}",
                '95th Percentile': f"${rwd_stats['p95']:,.0f}",
            }
        })
        st.dataframe(stats_df, use_container_width=True)

        vrp = rnd_stats['mean'] - rwd_stats['mean']
        st.caption(f"Variance risk premium (Q mean − P mean): **${vrp:+,.0f}**")

        if mu < 0:
            st.markdown(
                f"<div class='mu-warning'>⚠ Historical mu = {mu*100:.1f}% (trailing 365 days). "
                f"ETH has had a significant drawdown over this window, depressing the P-measure mean. "
                f"This reflects the lookback period, not a long-run expectation.</div>",
                unsafe_allow_html=True
            )

# 3D Surface

else:
    st.caption("Fetches all expiries sequentially — allow ~2 minutes.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate = st.button("Generate")

    if generate:
        today      = date.today()
        all_exp    = cached_expiries(asset)
        future_exp = [e for e in all_exp if (e - today).days > 35]

        progress = st.progress(0)
        status   = st.empty()
        results  = []

        for i, expiry in enumerate(future_exp):
            T_exp = (expiry - today).days / 365
            status.caption(f"Processing {expiry} ({i+1}/{len(future_exp)})...")
            try:
                df  = get_option_chain(str(expiry), currency=asset)
                rnd = compute_rnd(df, S=S, T=T_exp, r=r,
                                  strike_min=strike_min, strike_max=strike_max)
                rnd['days_to_expiry'] = (expiry - today).days
                results.append(rnd)
            except Exception:
                pass
            progress.progress((i + 1) / len(future_exp))

        status.empty()
        progress.empty()

        if len(results) < 2:
            st.error("Not enough expiries computed.")
            st.stop()

        days_list   = [r['days_to_expiry'].iloc[0] for r in results]
        strike_grid = results[0]['strike'].values
        Z           = np.array([r['density'].values for r in results])
        days_array  = np.array(days_list)
        days_fine   = np.linspace(days_array.min(), days_array.max(), 60)

        Z_interp = np.zeros((len(days_fine), len(strike_grid)))
        for i in range(len(strike_grid)):
            f              = interp1d(days_array, Z[:, i], kind='cubic')
            Z_interp[:, i] = f(days_fine)
        Z_interp = np.clip(Z_interp, 0, None)

        fig = go.Figure(data=[go.Surface(
            x=strike_grid, y=days_fine, z=Z_interp,
            name='RND Surface',
            colorscale='Viridis', reversescale=True,
            opacity=0.92,
            contours=dict(z=dict(show=True, usecolormap=True,
                                 highlightcolor='white', project_z=True))
        )])
        fig.add_scatter3d(
            x=[S, S], y=[min(days_list), max(days_list)], z=[0, 0],
            mode='lines', line=dict(color='#b03030', width=4),
            name=f'Current Price ${S:,.0f}'
        )
        fig.update_layout(
            title=f'{asset} Risk-Neutral Probability Density Surface · All Expiries',
            scene=dict(
                xaxis=dict(title='Price at Expiry ($)', tickformat='$,.0f',
                           gridcolor='#ddd', backgroundcolor=PAPER_BGCOLOR),
                yaxis=dict(title='Days to Expiry', gridcolor='#ddd',
                           backgroundcolor=PAPER_BGCOLOR, autorange='reversed'),
                zaxis=dict(title='Probability Density', gridcolor='#ddd',
                           backgroundcolor=PAPER_BGCOLOR),
                camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
                aspectratio=dict(x=1.5, y=1.2, z=0.8),
                bgcolor=PAPER_BGCOLOR,
            ),
            paper_bgcolor=PAPER_BGCOLOR,
            height=640,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Surface built from {len(results)} expiries.")

# Footer

st.markdown(
    "<div class='footer'>Data: Deribit · CoinGecko &nbsp;·&nbsp; © 2026</div>",
    unsafe_allow_html=True
)