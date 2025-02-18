import pandas as pd
import numpy as np

def calculate_derived_stats(df, close_col='close', high_col='high', low_col='low'):
    """
    Calculates derived financial statistics and adds them as new columns to the DataFrame.

    Args:
        df: pandas DataFrame containing OHLCV data.
        close_col: Name of the column containing closing prices (default: 'close').
        high_col: Name of the column containing high prices (default: 'high').
        low_col: Name of the column containing low prices (default: 'low').

    Returns:
        pandas DataFrame with added columns for derived statistics.  Returns None if
        there's an error (e.g., missing columns, insufficient data).
    """

    if not all(col in df.columns for col in [close_col, high_col, low_col]):
        print("Error: Missing required columns (Close, High, Low) in the input DataFrame.")
        return None

    # Check if there's enough data to calculate moving averages, etc.
    if len(df) < 20:
        print("Warning: Insufficient data for some calculations (e.g., 20-period SMA).  Results may be NaN.")


    # --- Exponential Moving Average (EMA) ---
    def calculate_ema(data, timeperiod):
        # pandas .ewm() is more numerically stable than a hand-rolled loop
        return data.ewm(span=timeperiod, adjust=False).mean()

    try:
        df['EMA_12'] = calculate_ema(df[close_col], 12)
        df['EMA_26'] = calculate_ema(df[close_col], 26)
    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return None

    # --- Moving Average Convergence Divergence (MACD) ---
    def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
        fast_ema = calculate_ema(data, fastperiod)
        slow_ema = calculate_ema(data, slowperiod)
        macd = fast_ema - slow_ema
        macd_signal = calculate_ema(macd, signalperiod)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    try:
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df[close_col])
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return None


    # --- Bollinger Bands ---
    def calculate_bollinger_bands(data, timeperiod=20):
        sma = data.rolling(window=timeperiod).mean()
        std = data.rolling(window=timeperiod).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        return upper_band, sma, lower_band

    try:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df[close_col])
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return None


    # --- Relative Strength Index (RSI) ---
    def calculate_rsi(data, timeperiod=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=timeperiod, min_periods=1).mean()  # Initial SMA
        avg_loss = loss.rolling(window=timeperiod, min_periods=1).mean()  # Initial SMA
        
        # Use .ewm() for a more precise and stable calculation after initial period.
        for i in range(timeperiod, len(data)):
          avg_gain[i] = (avg_gain[i-1] * (timeperiod - 1) + gain[i]) / timeperiod
          avg_loss[i] = (avg_loss[i-1] * (timeperiod - 1) + loss[i]) / timeperiod

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    try:
        df['RSI'] = calculate_rsi(df[close_col])
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None

    # --- Simple Moving Average (SMA) ---
    def calculate_sma(data, timeperiod):
        return data.rolling(window=timeperiod).mean()

    try:
        df['SMA_20'] = calculate_sma(df[close_col], 20)
        df['SMA_50'] = calculate_sma(df[close_col], 50)
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return None

    # --- Average True Range (ATR) ---
    def calculate_atr(high, low, close, timeperiod=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=timeperiod,min_periods=1).mean()

        # Exact calculation
        for i in range(timeperiod, len(atr)):
          atr[i] = (atr[i - 1] * (timeperiod - 1) + true_range[i]) / timeperiod

        return atr

    try:
        df['ATR'] = calculate_atr(df[high_col], df[low_col], df[close_col])
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return None

    # --- On Balance Volume (OBV) ---
    def calculate_obv(close, volume):
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)


    try:
        if 'volume' in df.columns:
            df['OBV'] = calculate_obv(df[close_col], df['volume'].astype(float))
        else:
            print("Warning: 'volume' column not found.  OBV will not be calculated.")
            df['OBV'] = np.nan
    except Exception as e:
        print(f"Error calculating OBV: {e}")
        return None

    return df

def process_csv(input_csv_path, output_csv_path):
    """
    Reads a CSV file, calculates derived stats, and writes the result to a new CSV file.

    Args:
        input_csv_path: Path to the input CSV file.
        output_csv_path: Path to the output CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Input CSV file is empty: {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Convert 'timestamp' to datetime objects, handling potential errors
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting 'timestamp' to datetime: {e}.  Proceeding without conversion.")
        # Depending on your needs, you might choose to exit here, or continue without the conversion.

    df = calculate_derived_stats(df)

    if df is not None:
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"Successfully processed and saved to {output_csv_path}")
        except Exception as e:
            print(f"Error writing to output CSV: {e}")

# Example usage (assuming you have 'input.csv' in the same directory):
input_file = 'raw_data/xrp.csv'
output_file = 'augmented_data/xrp.csv'
process_csv(input_file, output_file)