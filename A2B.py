"""
A2B.py
======
Command-line entry point for Assignment 2B route prediction.

Given an origin site, a destination site, a time of day, and a model
type, this script:
  1. Loads the chosen traffic-flow prediction model (RNN/LSTM/GRU).
  2. Predicts the traffic flow (vehicles/hour) at every site in the
     Boroondara network for the requested hour.
  3. Converts each road segment's predicted flow into a travel speed
     and time (including a 0.5 min junction delay per segment).
  4. Runs the A* search algorithm to find the fastest route from the
     origin to the destination.
  5. Prints the route and total driving time.
  6. Renders a Folium map highlighting the best route (saved to an
     HTML file and opened in your browser).

Usage:
    python A2B.py <origin_site> <destination_site> <time_HHMM> <model>

Example:
    python A2B.py 2000 2825 1100 RNN

    -> origin site      = 2000
    -> destination site = 2825
    -> time             = 1100 (11:00, i.e. hour index 11)
    -> model            = RNN  (loads models/rnn_model.keras etc.)

Sample output:
    Best path: 2000 –(412)- 3682 –(389)- 3127 –(401)- 4057 –(355)- 4032 –(298)- 2825
    Total Driving Time: 18.4 min

    Map saved and opened: route_2000_to_2825_11AM.html
"""

import sys

from config import COORDS, HOUR_LABELS
from model_utils import load_artefacts, load_long_data, predict_site_flows
from route_finder import find_best_route
from route_map import build_route_map, save_and_open_map


# ── Parse and validate command-line arguments ───────────────────────────────────
def parse_args():
    """
    Parses the four required positional arguments:
        origin_site, destination_site, time_HHMM, model_name

    Returns a tuple (origin, destination, hour, model_name).
    Exits with a usage message if the arguments are missing or invalid.
    """
    if len(sys.argv) != 5:
        print(
            "Usage: python A2B.py <origin_site> <destination_site> "
            "<time_HHMM> <model>\n"
            "Example: python A2B.py 2000 2825 1100 RNN"
        )
        sys.exit(1)

    origin_str, destination_str, time_str, model_name = sys.argv[1:5]

    # ── Site IDs ─────────────────────────────────────────────────────────────
    try:
        origin = int(origin_str)
        destination = int(destination_str)
    except ValueError:
        print(f"Error: origin and destination must be integer site IDs "
              f"(got '{origin_str}', '{destination_str}').")
        sys.exit(1)

    for site_id, label in [(origin, "Origin"), (destination, "Destination")]:
        if site_id not in COORDS:
            print(f"Error: {label} site {site_id} is not a known site. "
                  f"Valid sites are: {sorted(COORDS.keys())}")
            sys.exit(1)

    if origin == destination:
        print("Error: origin and destination sites must be different.")
        sys.exit(1)

    # ── Time -> hour index (HHMM format, e.g. 1100 -> hour 11) ──────────────────
    try:
        time_value = int(time_str)
    except ValueError:
        print(f"Error: time must be in HHMM format (e.g. 1100), got '{time_str}'.")
        sys.exit(1)

    hour = time_value // 100
    minute = time_value % 100

    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        print(f"Error: '{time_str}' is not a valid HHMM time.")
        sys.exit(1)

    return origin, destination, hour, model_name


# ── Format the final route output ────────────────────────────────────────────────
def format_route_output(path, edges, total_time):
    """
    Builds the two output lines:
        Best path: A –(flow)- B –(flow)- C ...
        Total Driving Time: X.X min
    """
    parts = [str(path[0])]
    for _, to_site, flow, _ in edges:
        parts.append(f"\u2013({flow})-")  # \u2013 = en-dash, matches "–"
        parts.append(str(to_site))

    best_path_line = "Best path: " + " ".join(parts)
    total_time_line = f"Total Driving Time: {total_time:.1f} min"

    return best_path_line, total_time_line


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    origin, destination, hour, model_name = parse_args()

    print(f"Origin site      : {origin}")
    print(f"Destination site : {destination}")
    print(f"Time             : {HOUR_LABELS[hour]} (hour index {hour})")
    print(f"Model            : {model_name.upper()}")

    # 1. Load the chosen model + scaler + label encoder
    try:
        model, scaler, le = load_artefacts(model_name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    # 2. Load and reshape the traffic dataset
    try:
        df_long = load_long_data()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    # 3. Predict the flow at every site for the requested hour
    site_flows = predict_site_flows(COORDS.keys(), hour, model, scaler, le, df_long)

    # 4. Find the fastest route
    path, edges, total_time = find_best_route(origin, destination, site_flows)

    if path is None:
        print(f"\nNo route found between site {origin} and site {destination}.")
        sys.exit(1)

    # 5. Print the result in the required format
    best_path_line, total_time_line = format_route_output(path, edges, total_time)

    print()
    print(best_path_line)
    print(total_time_line)

    # 6. Render a map highlighting the best route and open it in the browser
    m, output_path = build_route_map(
        origin, destination, path, edges, site_flows, hour, total_time
    )
    abs_path = save_and_open_map(m, output_path)
    print(f"\nMap saved and opened: {abs_path}")


if __name__ == '__main__':
    main()