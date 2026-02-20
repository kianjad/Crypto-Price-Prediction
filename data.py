import requests
import pandas as pd

def get_btc_instruments():
    """Get all active BTC option instruments from Deribit"""
    url = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {
        "currency": "BTC",
        "kind": "option"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data['result']

def get_available_expiries():
    """Return a clean list of available expiry dates"""
    instruments = get_btc_instruments()
    expiries = sorted(set(i['expiration_timestamp'] for i in instruments))
    expiry_dates = pd.to_datetime(expiries, unit='ms').date
    return expiry_dates

def get_option_chain(target_date_str):
    """
    Pull all options for a given expiry date and return as a DataFrame.
    target_date_str format: 'YYYY-MM-DD', e.g. '2026-03-27'
    """
    instruments = get_btc_instruments()
    target_date = pd.Timestamp(target_date_str).date()

    # Filter to options expiring on our target date
    filtered = [
        i for i in instruments
        if pd.to_datetime(i['expiration_timestamp'], unit='ms').date() == target_date
    ]

    print(f"Found {len(filtered)} options for {target_date_str}")

    # Pull market data for each instrument
    rows = []
    for inst in filtered:
        ticker_url = "https://www.deribit.com/api/v2/public/ticker"
        resp = requests.get(ticker_url, params={"instrument_name": inst['instrument_name']})
        ticker = resp.json()['result']

        rows.append({
            'instrument': inst['instrument_name'],
            'strike': inst['strike'],
            'option_type': inst['option_type'],  # 'call' or 'put'
            'implied_vol': ticker.get('mark_iv', None),
            'mark_price': ticker.get('mark_price', None),
            'bid': ticker.get('best_bid_price', None),
            'ask': ticker.get('best_ask_price', None),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['implied_vol'])
    df = df[df['implied_vol'] > 0]
    df = df.sort_values('strike').reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = get_option_chain('2026-03-27')
    print(df.to_string())