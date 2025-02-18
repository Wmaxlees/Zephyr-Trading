import csv
import argparse

def reverse_csv(input_file, output_file):
    """Reverses the order of rows in a CSV file.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
    """

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read and store the header row
            rows = list(reader)     # Read all rows into a list
            rows.reverse()         # Reverse the list of rows

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(rows)   # Write the reversed rows

        print(f"CSV file '{input_file}' reversed and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:  # Catch other potential errors (e.g., UnicodeDecodeError)
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse the order of rows in a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    reverse_csv(args.input_file, args.output_file)