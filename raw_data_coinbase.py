import ccxt
import pandas as pd
import datetime
import time

def get_coinbase_historical(symbol, start_date, end_date):
    """
    Retrieves historical open, close, and volume data for a given coin on Coinbase Pro.

    Args:
        symbol (str): Ticker symbol (e.g., 'BTC/USD', 'ETH/USD').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing historical OHLCV data, or None if
                      an error occurs.  Returns an empty DataFrame if no
                      data is found.o
    """
    coinbasepro = ccxt.coinbase()
    coinbasepro.load_markets()  # Load market data

    try:
        # Check if the symbol is valid on Coinbase Pro
        if symbol not in coinbasepro.symbols:
            print(f"Error: Symbol {symbol} not found on Coinbase Pro")
            return None

        ohlcv_data = []
        current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        # Coinbase Pro API uses 'granularity' instead of 'timeframe'
        # 1 day = 86400 seconds
        # 1 hour = 3600 seconds
        granularity = 3600

        while current_date <= end_datetime:
            # Coinbase Pro API expects timestamps in seconds (not milliseconds)
            start_timestamp = int(current_date.timestamp())
            end_timestamp = int((current_date + datetime.timedelta(days=1)).timestamp()) - 1 # End of the day

            try:
                # Fetch historical data.  Coinbase Pro API has a max of 300 candles per request.
                ohlcv = coinbasepro.fetch_ohlcv(symbol, timeframe='1h', since=start_timestamp * 1000, limit=300)

                if ohlcv:
                    ohlcv_data.extend(ohlcv)
                    # Update current_date. Add *number of candles returned* * granularity
                    # This is crucial for efficient and correct pagination.
                    current_date += datetime.timedelta(seconds=len(ohlcv) * granularity)

                else:
                    # No data for this day, move to the next
                    current_date += datetime.timedelta(hours=1)

            except ccxt.RequestTimeout:
                print(f"Request timeout for {symbol} on {current_date.strftime('%Y-%m-%d')}, retrying...")
                time.sleep(10)  # Wait before retrying
                continue

            except ccxt.ExchangeError as e:
                print(f"Exchange error for {symbol} on {current_date.strftime('%Y-%m-%d')}: {e}")
                if "429" in str(e): #rate limit
                    time.sleep(60)
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
        if not ohlcv_data:
            print(f"No data found for {symbol} between {start_date} and {end_date}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        print(f"Retrieved data for {symbol}")
        return df

    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None


# --- Example Usage ---
start_date = '2020-01-01'
end_date = '2024-12-31'

data = get_coinbase_historical('BTC/USD', start_date, end_date)
if data is not None and not data.empty:
    print(f"\nData:\n{data.head()}")
    data.to_csv("raw_data/btc.csv")

