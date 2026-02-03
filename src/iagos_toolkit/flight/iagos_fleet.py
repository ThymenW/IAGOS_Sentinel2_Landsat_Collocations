import csv
import os

def load_aircraft_csv(file_path):
    aircraft_data = {}
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            icao24 = row["icao24"]
            # Store all columns except icao24 (since it's the key)
            aircraft_data[icao24] = {key: value for key, value in row.items() if key != "icao24"}
    return aircraft_data

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
csv_path = os.path.join(current_dir, "aircraft_pars.csv")

# Load the aircraft data
AIRCRAFT_PARS = load_aircraft_csv(csv_path)
