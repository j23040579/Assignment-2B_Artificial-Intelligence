"""
model_utils.py
===============
Loads a trained traffic-flow model (RNN / LSTM / GRU — selected by name)
and predicts the traffic flow for a SINGLE target hour, for one or more
SCATS sites.

This is a leaner counterpart to the 24-hour-at-a-time prediction used by
the map-rendering script: the route finder only needs the flow at ONE
hour for every site that might appear on a path, so we avoid the cost of
predicting all 24 hours for every site.

The underlying sequence-construction / scaling / inverse-scaling logic
is identical to predict_all.py, so predictions stay consistent between
the map and the route finder.
"""

import importlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from config import MODEL_CONFIGS, DATA_PATH, SEQ_LENGTH, HOUR_COLUMNS


# ── Internal: map each model key to its training module filename ───────────────
# Must match the actual .py filenames in the project root.
_TRAINING_MODULES: dict = {
    "RNN":  "RNN",
    "LSTM": "LSTM",
    "GRU":  "GRU",
}


def _missing_artefacts(model_name: str) -> list:
    """
    Return a list of (key, path) pairs for every artefact in
    MODEL_CONFIGS[model_name] that does not yet exist on disk.
    Returns an empty list when all artefacts are present.
    """
    paths = MODEL_CONFIGS[model_name]
    return [(key, path) for key, path in paths.items() if not os.path.exists(path)]


def _run_training(model_name: str) -> None:
    """
    Dynamically import the training module for *model_name* and call its
    main() function to produce the model, scaler, and encoder artefacts.

    Adds the project root to sys.path if it is not already there so that
    RNN.py / LSTM.py / GRU.py can be imported regardless of how A2B.py
    was launched.

    Raises RuntimeError if:
      - the training module cannot be imported
      - main() raises an exception
      - one or more artefact files are still missing after training
    """
    module_name  = _TRAINING_MODULES[model_name]
    project_root = os.path.dirname(os.path.abspath(__file__))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Import the training script ─────────────────────────────────────────────
    try:
        training_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import training module '{module_name}.py'. "
            f"Make sure the file exists in the project root.\n"
            f"Original error: {exc}"
        ) from exc

    # ── Run the training pipeline ──────────────────────────────────────────────
    print(f"\n  Running {module_name}.main() — this may take a few minutes …")
    t0 = time.time()
    try:
        training_module.main()
    except Exception as exc:
        raise RuntimeError(
            f"Training pipeline for '{model_name}' raised an exception:\n{exc}"
        ) from exc

    elapsed = time.time() - t0

    # ── Verify artefacts were actually written ─────────────────────────────────
    still_missing = _missing_artefacts(model_name)
    if still_missing:
        missing_str = "\n".join(f"    ✗  {p}" for _, p in still_missing)
        raise RuntimeError(
            f"Training completed ({elapsed:.1f}s) but the following artefacts "
            f"are still missing:\n{missing_str}\n"
            f"Check {module_name}.py for errors."
        )

    print(f"  Training complete in {elapsed:.1f}s — artefacts saved:")
    for _, path in MODEL_CONFIGS[model_name].items():
        size_kb = os.path.getsize(path) / 1024
        print(f"    ✔  {path}  ({size_kb:.1f} KB)")


# ── 1. Load model + scaler + label encoder for the chosen model type ───────────
def load_artefacts(model_name):
    """
    Load the (model, scaler, label_encoder) triple for the given model
    name, e.g. 'RNN', 'LSTM', or 'GRU'.

    Auto-training
    -------------
    If one or more of the three artefact files (model, scaler, encoder)
    are missing from disk, the corresponding training script
    (RNN.py / LSTM.py / GRU.py) is imported and its main() is called
    automatically before the load is attempted again.  On subsequent
    runs the saved artefacts are reused and training is skipped.

    Raises
    ------
    ValueError        – model_name not in MODEL_CONFIGS
    RuntimeError      – training pipeline failed
    FileNotFoundError – artefacts still missing after training (safety net)
    """
    model_name = model_name.upper()

    if model_name not in MODEL_CONFIGS:
        valid = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model type '{model_name}'. Valid options are: {valid}"
        )

    # ── Check for missing artefacts; auto-train if any are absent ─────────────
    missing = _missing_artefacts(model_name)
    if missing:
        print(f"\n[model_utils] Artefacts missing for '{model_name}':")
        for key, path in missing:
            print(f"    ✗  {key}: {path}")
        print(f"[model_utils] Auto-training '{model_name}' now …")
        print("=" * 60)
        _run_training(model_name)
        print("=" * 60)

    paths = MODEL_CONFIGS[model_name]

    # Final guard — training should have created these, but be explicit.
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {key} for model '{model_name}' at '{path}'. "
                f"Check that the file exists and that you are running this "
                f"script from the project's root directory."
            )

    print(f"\nLoading '{model_name}' model artefacts...")
    model = load_model(paths['model_path'])

    with open(paths['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)

    with open(paths['encoder_path'], 'rb') as f:
        le = pickle.load(f)

    print(f"  Model   : {paths['model_path']}")
    print(f"  Scaler  : {paths['scaler_path']}")
    print(f"  Encoder : {paths['encoder_path']} ({len(le.classes_)} locations)")

    return model, scaler, le


# ── 2. Load and reshape the raw CSV into a long ('tidy') dataframe ─────────────
def load_long_data():
    """
    Reads Dataset/Time.csv (one row per SCATS site/location, one column
    per hour of the day) and reshapes it into a long-format dataframe
    with one row per (site, location, hour) combination.

    This is the same shape that predict_all.py uses, so the same scaler
    and label encoder apply without modification.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find traffic dataset at '{DATA_PATH}'. "
            f"Check that you are running this script from the project's "
            f"root directory."
        )

    df = pd.read_csv(DATA_PATH)

    rows = []
    for _, row in df.iterrows():
        for h, col in enumerate(HOUR_COLUMNS):
            rows.append({
                'scats_number':  row['SCATS Number'],
                'location':      row['Location'],
                'hour':          h,
                'flow_per_hour': row[col],
            })

    return pd.DataFrame(rows)


# ── 3. Predict the flow for ONE (site, direction) at ONE target hour ───────────
def predict_location_hour(site_id, location, target_hour, model, scaler, le, df_long):
    """
    Predict the traffic flow for a single approach/direction ('location')
    of a single SCATS site, at a single hour of the day (0-23).

    Returns an integer flow (vehicles/hour), or None if there isn't
    enough historical data to build a SEQ_LENGTH-step input sequence
    ending at target_hour.
    """
    if location not in le.classes_:
        return None

    location_enc = le.transform([location])[0]

    site_df = df_long[
        (df_long['scats_number'] == site_id) &
        (df_long['location'] == location)
    ].copy()

    if len(site_df) < SEQ_LENGTH + 1:
        return None

    site_df['location_enc'] = location_enc
    features = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data_scaled = scaler.transform(site_df[features].values)
    n_features = scaler.n_features_in_

    # Find the SEQ_LENGTH-step window whose NEXT row is target_hour,
    # then predict that next value.
    for i in range(len(data_scaled) - SEQ_LENGTH):
        if site_df.iloc[i + SEQ_LENGTH]['hour'] == target_hour:
            seq = data_scaled[i:i + SEQ_LENGTH].reshape(1, SEQ_LENGTH, n_features)
            pred_scaled = model.predict(seq, verbose=0)[0][0]

            # Inverse-transform: put the scaled prediction back into the
            # last ('flow_per_hour') column of a dummy row, then invert.
            dummy = np.zeros((1, n_features))
            dummy[0, -1] = pred_scaled
            predicted_flow = scaler.inverse_transform(dummy)[0, -1]

            return max(0, round(predicted_flow))

    return None


# ── 4. Predict the TOTAL flow for a whole site at ONE target hour ──────────────
def predict_site_flow(site_id, target_hour, model, scaler, le, df_long):
    """
    Sum the predicted flow across every approach/direction ('location')
    recorded for a given SCATS site, at a single hour of the day.

    This mirrors the way the map renderer aggregates per-direction
    predictions into a single per-site flow figure (see site_flow() in
    predict_all.py), but for one hour only.

    Returns 0 if the site has no usable directions for this hour.
    """
    site_rows = df_long[df_long['scats_number'] == site_id]
    locations = site_rows['location'].unique()

    total = 0
    found_any = False

    for location in locations:
        flow = predict_location_hour(
            site_id, location, target_hour, model, scaler, le, df_long
        )
        if flow is not None:
            total += flow
            found_any = True

    return total if found_any else 0


# ── 5. Predict the flow for a SET of sites at ONE target hour ──────────────────
def predict_site_flows(site_ids, target_hour, model, scaler, le, df_long):
    """
    Convenience wrapper: predicts predict_site_flow() for every site in
    site_ids and returns a dict {site_id: predicted_flow}.

    Used by the route finder to pre-compute the flow for every node in
    the road network once, before running A*.
    """
    print(f"\nPredicting flow at every site for hour index {target_hour}...")
    flows = {}
    for site_id in site_ids:
        flows[site_id] = predict_site_flow(site_id, target_hour, model, scaler, le, df_long)
        print(f"  Site {site_id}: {flows[site_id]:,} vehicles/hr")
    return flows


# ── 6. Predict ALL 24 HOURS for one (site, direction) ───────────────────────────
def predict_location_hours(site_id, location, model, scaler, le, df_long):
    """
    Predict the traffic flow for a single approach/direction ('location')
    of a single SCATS site, for EVERY hour of the day (0-23).

    This is used by the map renderer, which needs the full 24-hour
    profile for each site/direction to draw the hourly bar charts and
    compute daily totals / peak hours.

    Returns a list of 24 integer flows (vehicles/hour), or None if there
    isn't enough historical data to build SEQ_LENGTH-step sequences for
    this site/direction.
    """
    if location not in le.classes_:
        return None

    location_enc = le.transform([location])[0]
    site_df = df_long[
        (df_long['scats_number'] == site_id) &
        (df_long['location'] == location)
    ].copy()

    if len(site_df) < SEQ_LENGTH + 24:
        return None

    site_df['location_enc'] = location_enc
    features = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data_scaled = scaler.transform(site_df[features].values)
    n_features = scaler.n_features_in_
    predictions = []

    for target_hour in range(24):
        found = False
        for i in range(len(data_scaled) - SEQ_LENGTH):
            if site_df.iloc[i + SEQ_LENGTH]['hour'] == target_hour:
                seq = data_scaled[i:i + SEQ_LENGTH].reshape(1, SEQ_LENGTH, n_features)
                pred_scaled = model.predict(seq, verbose=0)[0][0]
                dummy = np.zeros((1, n_features))
                dummy[0, -1] = pred_scaled
                predicted_flow = scaler.inverse_transform(dummy)[0, -1]
                predictions.append(max(0, round(predicted_flow)))
                found = True
                break
        if not found:
            predictions.append(0)

    return predictions


# ── 7. Predict every site + direction, for all 24 hours ────────────────────────
def predict_all(model, scaler, le, df_long):
    """
    Run predict_location_hours() for every (site, direction) combination
    found in df_long.

    Returns a nested dict:
        { site_id: { location: [24 hourly predictions], ... }, ... }
    """
    site_locations = df_long.groupby('scats_number')['location'].unique()
    results = {}
    total = sum(len(locs) for locs in site_locations)
    done = 0

    print(f"\nPredicting {total} location-directions across {len(site_locations)} sites...")
    for site_id, locations in site_locations.items():
        results[site_id] = {}
        for location in locations:
            preds = predict_location_hours(site_id, location, model, scaler, le, df_long)
            if preds:
                results[site_id][location] = preds
            done += 1
            print(f"  [{done}/{total}] {site_id} | {location}")

    return results


# ── 8. Get flow for a site at a specific hour (or daily total) ─────────────────
def site_flow(site_id, results, hour=None):
    """
    Look up the predicted flow for a site from the full predict_all()
    results.

    hour=None -> sum all 24 hours across all directions (daily total)
    hour=13   -> sum only hour 13 (1PM) across all directions
    """
    if site_id not in results:
        return 0
    all_dirs = list(results[site_id].values())
    if not all_dirs:
        return 0
    if hour is None:
        return sum(sum(d) for d in all_dirs)
    else:
        return sum(d[hour] for d in all_dirs)