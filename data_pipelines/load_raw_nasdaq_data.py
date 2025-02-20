import csv
import sqlite3
import datetime
import os

DB_NAME = 'data.db'
TABLE_NAME = 'ohlcv_data'
EXCHANGE = 'NASDAQ'

def convert_date_to_timestamp(date_str):
    """Converts date string (MM/DD/YYYY) to integer timestamp."""
    date_object = datetime.datetime.strptime(date_str, '%m/%d/%Y')
    return int(date_object.timestamp())

def process_csv_file(csv_filepath, db_conn):
    """Processes a single CSV file and inserts data into the database."""
    symbol = os.path.splitext(os.path.basename(csv_filepath))[0].upper() # Extract symbol from filename, e.g., aapl from aapl.csv

    cursor = db_conn.cursor()

    try:
        with open(csv_filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp = convert_date_to_timestamp(row['Date'])
                open_price = float(row['Open'].replace('$', ''))
                high_price = float(row['High'].replace('$', ''))
                low_price = float(row['Low'].replace('$', ''))
                close_price = float(row['Close/Last'].replace('$', ''))
                volume = float(row['Volume'].replace(',', '')) # Remove comma for thousands separator if present

                sql = f"""
                    INSERT OR IGNORE INTO {TABLE_NAME} (timestamp, symbol, exchange, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(sql, (timestamp, symbol, EXCHANGE, open_price, high_price, low_price, close_price, volume))
    except Exception as e:
        print(f"Error processing file {csv_filepath}: {e}")
    finally:
        cursor.close()

def main():
    """Main function to connect to the database and process CSV files."""
    try:
        db_conn = sqlite3.connect(DB_NAME)
        print(f"Connected to database: {DB_NAME}")

        csv_files = ['raw_data/amd.csv', 'raw_data/amzn.csv', 'raw_data/csco.csv', 'raw_data/meta.csv', 'raw_data/msft.csv', 'raw_data/qcom.csv', 'raw_data/sbux.csv']
        if not csv_files:
            print("No CSV files found in the current directory.")
            return

        print(f"Found CSV files: {csv_files}")

        for csv_file in csv_files:
            print(f"Processing: {csv_file}")
            process_csv_file(csv_file, db_conn)
            print(f"Finished processing: {csv_file}")

        db_conn.commit()
        print("Data import complete.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()