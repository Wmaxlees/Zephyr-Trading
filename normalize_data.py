import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_financial_data(df, fill_method='mean', diff_order=1):
    """
    Preprocesses financial data by handling NaNs, applying log transformations,
    differencing, and standardization.

    Args:
        df (pd.DataFrame): The input DataFrame.  Must contain at least 'close',
                           and 'volume' columns.  Can contain other columns
                           which will be treated as indicators.
        fill_method (str, optional): Method for filling NaNs.  Options:
            'ffill' (forward fill, default), 'bfill' (backward fill),
            'mean' (fill with mean of the column), 'median' (fill with
            median), 'interpolate' (linear interpolation), None (no filling).
        diff_order (int, optional): The order of differencing to apply to the
            log prices. Defaults to 1.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    df = df.copy()

    # Rename columns to lowercase and remove spaces for consistency.
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # --- Handle Missing Values ---
    if fill_method == 'ffill':
        df = df.fillna(method='ffill')
    elif fill_method == 'bfill':
        df = df.fillna(method='bfill')
    elif fill_method == 'mean':
        df = df.fillna(df.mean())
    elif fill_method == 'median':
        df = df.fillna(df.median())
    elif fill_method == 'interpolate':
        df = df.interpolate()
    elif fill_method is None:
        pass  # Don't fill NaNs
    else:
        raise ValueError("Invalid fill_method.  Choose from 'ffill', 'bfill', "
                         "'mean', 'median', 'interpolate', or None.")

    # Check for remaining NaNs (after fill, or if fill_method=None)
    if df.isnull().values.any():
        remaining_nan_counts = df.isnull().sum()
        raise ValueError(f"NaNs still present after filling.  Remaining NaN counts per column:\n{remaining_nan_counts}")

    price_cols = ['open', 'high', 'low', 'close']
    for price_col in price_cols:
        log_price = np.log(df[price_col])
        diff = log_price.diff(periods=diff_order)
        scaler = StandardScaler()
        df[f'{price_col}_norm'] = scaler.fit_transform(diff.values.reshape(-1, 1))

    # --- Log Transformation (Volume) ---
    log_volume = np.log(df['volume'])
    scaler = StandardScaler()
    df['volume_norm'] = scaler.fit_transform(log_volume.values.reshape(-1, 1))

    # --- Standardization ---
    indicator_cols = [col for col in df.columns
                      if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for indicator_col in indicator_cols:
        scaler = StandardScaler()
        df[f'{indicator_col}_norm'] = scaler.fit_transform(df[indicator_col].values.reshape(-1, 1))

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

    df = preprocess_financial_data(df)

    if df is not None:
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"Successfully processed and saved to {output_csv_path}")
        except Exception as e:
            print(f"Error writing to output CSV: {e}")


# --- Example Usage (with provided data) ---
if __name__ == '__main__':
    input_file = 'augmented_data/btc.csv'
    output_file = 'normalized_data/btc.csv'

    process_csv(input_file, output_file)



