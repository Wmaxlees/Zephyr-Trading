import sqlite3

def flatten_and_write_ohlcv_data(db_file, new_table_name):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get unique symbols
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv_data")
    symbols = [row for row in cursor.fetchall()]

    # Construct the SQL query dynamically
    select_clauses = []
    for symbol in symbols:
        select_clauses.append(f"MAX(CASE WHEN symbol = '{symbol}' THEN close ELSE NULL END) AS {symbol}_close")
        select_clauses.append(f"MAX(CASE WHEN symbol = '{symbol}' THEN volume ELSE NULL END) AS {symbol}_volume")
        # Add more cases for other columns (open, high, low) as needed

    sql_query = f"""
        SELECT
            timestamp,
            {', '.join(select_clauses)}
        FROM ohlcv_data
        GROUP BY timestamp;
    """

    # Create the new table
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {new_table_name} (
            timestamp INTEGER PRIMARY KEY,
            {' REAL, '.join([f'{symbol}_close' for symbol in symbols])} REAL,
            {' REAL, '.join([f'{symbol}_volume' for symbol in symbols])} REAL
        )
    """
    cursor.execute(create_table_query)

    # Execute the query and fetch the results
    cursor.execute(sql_query)
    rows = cursor.fetchall()

    # Insert the flattened data into the new table
    insert_query = f"""
        INSERT INTO {new_table_name} (timestamp, {', '.join([f'{symbol}_close' for symbol in symbols])}, {', '.join([f'{symbol}_volume' for symbol in symbols])})
        VALUES ({', '.join(['?'] * (len(symbols) * 2 + 1))})
    """
    cursor.executemany(insert_query, rows)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Example usage
db_file = 'data.db'
new_table_name = 'flattened_ohlcv'
flatten_and_write_ohlcv_data(db_file, new_table_name)