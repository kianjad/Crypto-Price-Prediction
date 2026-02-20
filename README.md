# Crypto Price Prediction

Extracts the **market-implied probability distribution** of future Bitcoin and Ethereum prices from live Deribit options data using the Breeden-Litzenberger theorem. Compares these risk-neutral (Q-measure) densities against real-world (P-measure) lognormal densities estimated from historical returns, and visualizes the full term structure as an interactive 3D surface across all expiries.

---

## What This Project Does

Options markets encode the collective beliefs of market participants about future asset prices. By observing how implied volatility varies across strikes — the **volatility smile** — we can recover the full probability distribution the market is pricing in, without assuming any particular model.

This project does that in three steps:

1. **Pulls the live option chain** from Deribit's public API for BTC and ETH across all available expiries
2. **Applies Breeden-Litzenberger** to extract the risk-neutral density (Q-measure) for each expiry
3. **Estimates the real-world density** (P-measure) from historical log-returns via CoinGecko, and compares it to the market-implied one

The gap between Q and P is economically meaningful — it reflects the **variance risk premium** and crash risk aversion embedded in options prices.

---

## The Math

### Breeden-Litzenberger (1978)

The risk-neutral density $q(K)$ of the underlying price at expiry $T$ can be recovered from the second derivative of call prices with respect to strike:

$$q(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$$

where $C(K)$ is the price of a European call with strike $K$.

**Implementation:**
- Filter to OTM options (calls above spot, puts below spot)
- Fit a degree-4 smoothing spline to implied volatility as a function of strike
- Convert smoothed IV → call prices via Black-Scholes
- Differentiate twice using a second spline → raw density
- Normalize so the density integrates to 1

### Black-Scholes Call Pricing

$$C = S \cdot \Phi(d_1) - K e^{-rT} \cdot \Phi(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Real-World (P-measure) Density

Under the lognormal assumption, $\log(S_T/S_0) \sim \mathcal{N}(\mu T, \sigma^2 T)$ where $\mu$ and $\sigma$ are estimated from daily historical log-returns and annualized. This gives a lognormal density:

$$f(K) = \frac{1}{K \sigma_{\log} \sqrt{2\pi}} \exp\left(-\frac{(\ln K - \mu_{\log})^2}{2\sigma_{\log}^2}\right)$$

where $\mu_{\log} = \ln S + \mu T$ and $\sigma_{\log} = \sigma\sqrt{T}$.

---

## Results

### BTC — Mar 27 2026 Expiry

| Statistic | Risk-Neutral (Q) | Real-World (P) |
|-----------|-----------------|----------------|
| Mean      | $68,090         | $65,564        |
| Std Dev   | $9,975          | $9,561         |
| Skewness  | 0.23            | 0.45           |
| Excess Kurtosis | 0.77      | 0.34           |
| 5th Percentile | $51,824    | $51,222        |
| 95th Percentile | $84,689   | $82,485        |

The Q mean sits ~$2,500 above the P mean — this spread is the **variance risk premium**: options buyers pay a premium for protection, so market-implied expectations are systematically higher than historical drift would suggest. The Q distribution also has a fatter right tail (higher kurtosis), reflecting the premium on upside calls.

### ETH — Mar 27 2026 Expiry

| Statistic | Risk-Neutral (Q) | Real-World (P) |
|-----------|-----------------|----------------|
| Mean      | $1,973          | $1,953         |
| Std Dev   | $408            | $480           |
| Skewness  | 0.25            | 0.67           |
| Excess Kurtosis | 0.66      | 0.56           |
| 5th Percentile | $1,313     | $1,275         |
| 95th Percentile | $2,660    | $2,833         |

ETH shows an inverted pattern vs BTC: the P distribution is wider than Q. This is driven by a negative historical drift (annualized mu ≈ -33%) over the trailing year due to ETH's significant drawdown, which widens the P tails while pulling its mode left. The Q distribution, anchored to current options pricing, is tighter and more centered.

---

## Project Structure

```
├── data.py          # Pulls BTC option chain from Deribit API
├── density.py       # Breeden-Litzenberger implementation + Black-Scholes pricer
├── historical.py    # Fetches historical prices from CoinGecko, estimates mu/sigma
├── plot.py          # Single-expiry BTC RND plot
├── compare.py       # BTC Q vs P comparison plot + summary statistics
├── compare_eth.py   # ETH Q vs P comparison plot + summary statistics
├── surface.py       # BTC 3D density surface across all expiries
└── surface_eth.py   # ETH 3D density surface across all expiries
```

---

## Installation

```bash
git clone https://github.com/yourusername/crypto-rnd
cd crypto-rnd
pip install numpy pandas scipy plotly requests
```

---

## Usage

All scripts fetch live data at runtime — no data files needed.

```bash
# Single-expiry BTC density
python plot.py

# BTC: risk-neutral vs real-world comparison
python compare.py

# ETH: risk-neutral vs real-world comparison
python compare_eth.py

# BTC 3D density surface (all expiries — takes ~2 min)
python surface.py

# ETH 3D density surface (all expiries — takes ~2 min)
python surface_eth.py
```

Spot prices are fetched live from Deribit's index price API. Historical prices are fetched from CoinGecko's free tier (365-day limit).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `scipy` | Spline fitting, normal distribution |
| `plotly` | Interactive visualization |
| `requests` | API calls to Deribit and CoinGecko |

---

## Key Design Decisions

**OTM-only options for vol surface fitting:** Deep ITM options have wide bid-ask spreads and low liquidity. Using only OTM calls (above spot) and OTM puts (below spot) gives a cleaner, more reliable vol smile to differentiate.

**Spline smoothing parameter:** The smoothing factor `s = len(strikes) * 0.001` balances capturing the smile shape without overfitting to noisy quotes. The RND is sensitive to this — too stiff loses the smile, too loose introduces oscillation artifacts in the tails.

**Normalization:** The raw second derivative is normalized to integrate to 1 over the strike range, making the densities directly comparable across expiries and assets.

**Live spot price:** Spot is fetched from Deribit's `get_index_price` endpoint at runtime so the density and annotations always reflect current market conditions.

---

## Limitations

- **Strike range truncation:** Density is computed over a finite strike range ($40k–$140k for BTC, $800–$4,000 for ETH). Probability mass in the tails beyond this range is lost, so densities are slightly under-normalized at the extremes.
- **Free API limits:** CoinGecko's free tier limits historical data to 365 days, which may not capture full market cycles. ETH's negative trailing mu reflects its recent drawdown period specifically.
- **No microstructure correction:** Bid-ask spreads are not accounted for — mark IV is used directly. In illiquid strikes this can introduce noise.
- **Deribit-specific:** Results reflect the Deribit options market specifically. Other venues may price risk differently.

---

## References

- Breeden, D. T., & Litzenberger, R. H. (1978). *Prices of state-contingent claims implicit in option prices.* Journal of Business, 51(4), 621–651.
- Black, F., & Scholes, M. (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
- Deribit API Documentation: https://docs.deribit.com
- CoinGecko API Documentation: https://www.coingecko.com/en/api
