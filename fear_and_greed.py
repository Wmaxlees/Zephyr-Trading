import requests
import csv
import os
from datetime import datetime, date, timedelta
import io  # For handling string data as files

def fetch_alternative_me_fear_greed_api(start_date):
    """
    Fetches historical Fear and Greed Index data from Alternative.me API.

    Args:
        start_date (str): Start date in YYYY-MM-DD format (e.g., "2020-01-01").

    Returns:
        list: A list of dictionaries, where each dictionary represents a day's
              Fear and Greed Index data, or None if there was an error.
    """
    base_url = "https://api.alternative.me/fng/"
    format_param = "csv"
    date_format_param = "us"  # MM-DD-YYYY format

    data = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = date.today()

    time_difference = end_date - current_date
    days_difference = time_difference.days

    url = f"{base_url}?limit={days_difference}&format={format_param}&date_format={date_format_param}"
    print(url)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses

        print(response.text)

        csv_text = response.text

        # Use io.StringIO to treat the string as a file
        csvfile = io.StringIO(csv_text)
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            data.append({
                'date': row['date'],
                'fng_classification': row['fng_classification'],
                'fng_value': row['fng_value']
            })


    except requests.exceptions.RequestException as e:
        print(f"Request error for {current_date.strftime('%Y-%m-%d')}: {e}")
        return None

    return data

def write_to_csv(data, filename="alternative_me_fear_greed_index_api.csv"):
    """
    Writes Fear and Greed Index data to a CSV file.

    Args:
        data (list): List of dictionaries containing Fear and Greed Index data.
        filename (str): Name of the CSV file to write to.
    """
    if not data:
        print("No data to write to CSV.")
        return

    fieldnames = data[0].keys() if data else []
    if not fieldnames:
        print("No fields found in data.")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data written to {filename}")


def main():
    start_date = "2020-01-01" # February 17, 2020
    fear_greed_data = fetch_alternative_me_fear_greed_api(start_date)

    if fear_greed_data:
        write_to_csv(fear_greed_data, filename="alternative_me_fear_greed_index_api.csv")
    else:
        print("Failed to fetch Fear and Greed Index data from Alternative.me API.")

if __name__ == "__main__":
    main()