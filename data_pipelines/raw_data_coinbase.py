import ccxt
import pandas as pd
import datetime
import time
import sqlite3
from datetime import date, timedelta

def create_sqlite_connection(db_file):
    """ Create a database connection to the SQLite database specified
        by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_ohlcv_table(conn):
    """ Create a single table to store OHLCV data for all symbols and exchanges if it doesn't exist.
    :param conn: Connection object
    """
    try:
        cursor = conn.cursor()
        table_name = 'ohlcv_data' # Generic table name for all OHLCV data
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp INTEGER,
                symbol TEXT,
                exchange TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (timestamp, symbol, exchange)
            )
        ''')
        conn.commit()
        print(f"Table '{table_name}' created or already exists.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def insert_ohlcv_data(conn, symbol, exchange, df):
    """ Insert OHLCV data from DataFrame into SQLite table.
    :param conn: Connection object
    :param symbol: Ticker symbol
    :param exchange: Exchange name
    :param df: Pandas DataFrame containing OHLCV data
    """
    table_name = 'ohlcv_data' # Generic table name
    try:
        cursor = conn.cursor()
        for index, row in df.iterrows():
            timestamp_ms = int(index.timestamp() * 1000) # Convert Timestamp to milliseconds for storage if needed, otherwise just use seconds int(index.timestamp())
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table_name} (timestamp, symbol, exchange, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp_ms, symbol, exchange, row['open'], row['high'], row['low'], row['close'], row['volume'])) # Using milliseconds for timestamp in DB
        conn.commit()
        print(f"Inserted {len(df)} records into table '{table_name}' for symbol '{symbol}' from '{exchange}'.")
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")


def get_coinbase_historical(symbol, start_date, end_date, db_name="crypto_data.db"):
    """
    Retrieves historical open, close, and volume data for a given coin on Coinbase Pro
    and saves it to a single SQLite database table.

    Args:
        symbol (str): Ticker symbol (e.g., 'BTC/USD', 'ETH/USD').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        db_name (str): Name of the SQLite database file.

    Returns:
        pd.DataFrame: A DataFrame containing historical OHLCV data, or None if
                      an error occurs. Returns an empty DataFrame if no
                      data is found.
    """
    exchange_name = "coinbase" # Define exchange name here
    coinbasepro = ccxt.coinbase()
    coinbasepro.load_markets()  # Load market data

    conn = create_sqlite_connection(db_name)
    if conn is None:
        return None

    # Create the table once when the script starts, not per symbol
    # create_ohlcv_table(conn, symbol) # No longer need to create table per symbol, create once outside if needed, or let the first run create it.

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
        granularity = 86400

        while current_date <= end_datetime:
            print(f"Fetching data for {symbol} on {current_date.strftime('%Y-%m-%d')}...")

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
                    current_date += datetime.timedelta(days=1)

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
        print(f"Retrieved data for {symbol} from {exchange_name}")

        insert_ohlcv_data(conn, symbol, exchange_name, df) # Insert data to SQLite, passing exchange name
        conn.close() # Close connection after inserting data
        return df

    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None


# --- Example Usage ---
# Database and date setup (create table only once if needed)
database_name = "data.db"
conn = create_sqlite_connection(database_name) # Create connection outside to manage table creation once
if conn:
    create_ohlcv_table(conn) # Create the table if it doesn't exist - only needs to run once
    conn.close() # Close connection after table creation

start_date = '2020-01-01'
yesterday = date.today() - timedelta(days=1)
end_date = yesterday.strftime('%Y-%m-%d') # Format to 'YYYY-MM-DD' string


# Fetch and store data for multiple symbols
symbols = ['SOL/USD', 'BTC/USD', 'ETH/USD', 'XRP/USD'] # Example symbols
for symbol in symbols:
    print(f"\nFetching data for {symbol}...")
    data = get_coinbase_historical(symbol, start_date, end_date, database_name)
    if data is not None and not data.empty:
        print(f"\nData for {symbol}:\n{data.head()}")
        print(f"Data for {symbol} saved to SQLite database '{database_name}'.")

print(f"Data ends on: {end_date}") # Print the dynamic end date