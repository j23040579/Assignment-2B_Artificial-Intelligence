"""
A2B.py
======
Command-line entry point for Assignment 2B route prediction.

Given an origin site, a destination site, a time of day, and a model
type, this script:
  1. Loads the chosen traffic-flow prediction model (RNN/LSTM/GRU).
     *** If the model has never been trained, it is trained automatically
         before loading — no manual step required. ***
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
    python A2B_3routes.py <origin_site> <destination_site> <time_HHMM> <model>

Example:
    python A2B.py 2000 2825 1100 RNN

    -> origin site      = 2000
    -> destination site = 2825
    -> time             = 1100 (11:00, i.e. hour index 11)
    -> model            = RNN  (auto-trains if models/rnn_model.keras is missing)

Auto-training
-------------
If the requested model's artefacts (model file + scaler + label encoder)
are not found on disk, A2B.py will automatically run the corresponding
training script (RNN.py / LSTM.py / GRU.py) and save the artefacts before
continuing with route finding.  On subsequent runs the saved artefacts are
reused and training is skipped.

Sample output:
    Best path: 2000 –(202)- 3682 –(156)- 3127 –(108)- 4057 –(83)- 4032 –(180)- 2825
    Total Driving Time: 16.3 min

    2nd path: 2000 –(202)- 3682 –(156)- 3127 –(192)- 3120 –(195)- 4032 –(180)- 2825
    Total Driving Time: 17.5 min

    3rd path: 2000 –(174)- 4043 –(200)- 3120 –(195)- 4032 –(180)- 2825
    Total Driving Time: 17.8 min

    Route 1 map saved and opened: route_2000_to_2825_11AM.html
    Route 2 map saved and opened: route_2000_to_2825_11AM.html
    Route 3 map saved and opened: route_2000_to_2825_11AM.html
"""

import sys

from config import COORDS, HOUR_LABELS
from model_utils import load_artefacts, load_long_data, predict_site_flows
from route_finder import find_top_3_routes
from route_map import (build_route_map, save_and_open_map)


# ── Parse and validate command-line arguments ─────────────────────────────────
def parse_args():
    """
    Parses the four required positional arguments:
        origin_site, destination_site, time_HHMM, model_name

    Returns a tuple (origin, destination, hour, model_name).
    Exits with a usage message if the arguments are missing or invalid.
    """
    if len(sys.argv) != 5:
        print(
            "Usage: python A2B.py <origin> <destination> <time_HHMM> <model>"
        )
        sys.exit(1)

    origin_str, destination_str, time_str, model_name = sys.argv[1:5]

    # ── Validate model name early so we don't waste time parsing coordinates ──
    supported_models = ("RNN", "LSTM", "GRU")
    if model_name.upper() not in supported_models:
        print(
            f"Error: unknown model '{model_name}'.\n"
            f"Supported models: {', '.join(supported_models)}"
        )
        sys.exit(1)
    # ── Site IDs ──────────────────────────────────────────────────────────────
    try:
        origin      = int(origin_str)
        destination = int(destination_str)
    except ValueError:
        print(f"Error: origin and destination must be integer site IDs "
              f"(got '{origin_str}', '{destination_str}').")
        sys.exit(1)

    for site_id, label in [(origin, "Origin"), (destination, "Destination")]:
        if site_id not in COORDS:
            print(
                f"Error: {label} site {site_id} is not a known site.\n"
                f"Valid sites: {sorted(COORDS.keys())}")
            sys.exit(1)

    if origin == destination:
        print("Error: origin and destination sites must be different.")
        sys.exit(1)

    # ── Time -> hour index (HHMM format, e.g. 1100 -> hour 11) ───────────────
    try:
        time_value = int(time_str)
    except ValueError:
        print(
            f"Error: time must be in HHMM format (e.g. 1100), got '{time_str}'."
            )
        sys.exit(1)

    hour   = time_value // 100
    minute = time_value % 100

    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        print(f"Error: '{time_str}' is not a valid HHMM time.")
        sys.exit(1)

    return origin, destination, hour, model_name.upper()


# ── Format the final route output ─────────────────────────────────────────────
def format_route_output(rank, path, edges, total_time):
    """
    Builds the two output lines:
        Best path: A –(flow)- B –(flow)- C ...
        Total Driving Time: X.X min
    """
    rank_labels = {
        1: "Best path", 
        2: "2nd path", 
        3: "3rd path"
    }
    label = rank_labels.get(rank, f"Route {rank}")

    parts = [str(path[0])]
    for _, to_site, flow, _ in edges:
        parts.append(f"\u2013({flow})-")   # en-dash
        parts.append(str(to_site))

    path_line = f"{label}: " + " ".join(parts)
    time_line = f"Total Driving Time: {total_time:.1f} min"

    return path_line, time_line


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    origin, destination, hour, model_name = parse_args()

    print(f"\nOrigin site      : {origin}")
    print(f"Destination site : {destination}")
    print(f"Time             : {HOUR_LABELS[hour]} (hour index {hour})")
    print(f"Model            : {model_name}")

    # ── Step 1: Load (or auto-train) the chosen model ─────────────────────────
    # load_artefacts() checks whether the model file, scaler, and label encoder
    # all exist.  If any are missing it calls train_runner.train_model() first,
    # then loads the freshly-saved artefacts.
    print(f"\n[Step 1/4] Loading {model_name} model …")
    try:
        model, scaler, le = load_artefacts(model_name)
    except ValueError as exc:
        # Unrecognised model name (already caught in parse_args, but be safe).
        print(f"\nError: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        # Training ran but artefacts are still missing — something went wrong.
        print(f"\nError: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        # The training pipeline itself raised an exception.
        print(f"\nTraining failed:\n{exc}")
        sys.exit(1)

    # ── Step 2: Load the traffic dataset ──────────────────────────────────────
    print("\n[Step 2/4] Loading traffic dataset ...")
    try:
        df_long = load_long_data()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    # ── Step 3: Predict flows at every site for the requested hour ─────────────
    print(f"\n[Step 3/4] Predicting flows for hour {hour} ({HOUR_LABELS[hour]}) ...")
    site_flows = predict_site_flows(
        COORDS.keys(), hour, model, scaler, le, df_long
    )

    # Step 4: Find top 3 routes with A*
    print(f"\n[Step 4/4] Running A* search: {origin} → {destination} ...")
    routes = find_top_3_routes(origin, destination, site_flows)

    print("\nDEBUG:")  
    for rank, (path, edges, total_time) in enumerate(routes, start=1): 
        print(f"Route {rank}:") 
        print(path) 
        print(edges) 
        print()

    if not routes:
        print(f"\nNo route found between site {origin} and site {destination}.")
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    print()
    for rank, (path, edges, total_time) in enumerate(routes, start=1):
        path_line, time_line = format_route_output(rank, path, edges, total_time)
        print(path_line)
        print(time_line)
        if rank < len(routes):
            print()

    # ── Generate ONE map with 3 routes ───────────────────────────────
    print("\nGenerating map...")

    path1, edges1, total_time1 = routes[0]

    # Route 2 and Route 3 (if found) are overlaid on the same map.
    extra_routes = routes[1:]

    m, output_path = build_route_map(
        origin,
        destination,
        path1,
        edges1,
        site_flows,
        hour,
        total_time1,
        route_number=1,
        extra_routes=extra_routes,
    )

    abs_path = save_and_open_map(
        m,
        output_path
    )

    print(
        f"\nAll routes map saved and opened: {abs_path}"
    )


if __name__ == "__main__":
    main()