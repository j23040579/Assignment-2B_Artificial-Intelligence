"""
Traffic Prediction using SimpleRNN (TensorFlow/Keras)
======================================================
Dataset  : SCATS traffic data  (Time.csv)
Goal     : Predict the next hour's vehicle flow given the previous
           SEQ_LENGTH hours, for a specific SCATS site AND direction
           (Location string).

IMPORTANT: SEQ_LENGTH, DATA_PATH, EPOCHS, BATCH_SIZE, and TEST_SPLIT are
all imported from config.py instead of being hardcoded here. This is
critical because model_utils.py (used by A2B.py / predict_map.py at
INFERENCE time) also imports SEQ_LENGTH from config.py — if this
training script used a different value, the saved model would expect a
different input shape than what's fed to it later, causing a shape
mismatch or silently wrong predictions.

To change the sequence length, batch size, etc. for ALL three models at
once, edit config.py — do not edit the values here.

Key change vs original:
  - 'Location' is now encoded and included as a feature alongside
    scats_number, hour, and flow_per_hour.
  - test_prediction() now accepts a `location` string so you can target
    a specific approach direction at a site.
  - main() now checks if all model artefacts (.keras + .pkl files) already
    exist on disk; if they do, training is skipped and the saved files are
    loaded directly. Only when at least one artefact is missing will the
    full training pipeline run.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    SEQ_LENGTH, DATA_PATH, EPOCHS, BATCH_SIZE, TEST_SPLIT,
    MODEL_CONFIGS, HOUR_COLUMNS,
)

# ── Configuration (sourced from config.py — see note above) ────────────────────
MODEL_SAVE_PATH   = MODEL_CONFIGS['RNN']['model_path']
SCALER_SAVE_PATH  = MODEL_CONFIGS['RNN']['scaler_path']
ENCODER_SAVE_PATH = MODEL_CONFIGS['RNN']['encoder_path']

# All artefacts that must be present to skip training
REQUIRED_ARTEFACTS = [MODEL_SAVE_PATH, SCALER_SAVE_PATH, ENCODER_SAVE_PATH]


# ── 0. Artefact check ────────────────────────────────────────────────────────
def all_artefacts_exist() -> bool:
    """
    Return True only when every required model artefact exists on disk.
    If even one file is missing, training must run so the set is complete.
    """
    missing = [p for p in REQUIRED_ARTEFACTS if not os.path.isfile(p)]
    if missing:
        print("\n The following artefacts are missing - training will run:")
        for p in missing:
            print(f"   x  {p}")
        return False

    print("\n All model artefacts found - skipping training:")
    for p in REQUIRED_ARTEFACTS:
        size_kb = os.path.getsize(p) / 1024
        print(f"   ok {p}  ({size_kb:.1f} KB)")
    return True


# ── Helper: load saved artefacts ────────────────────────────────────────────────
def load_artefacts():
    """Load and return (model, scaler, label_encoder) from disk."""
    print("\n Loading saved artefacts ...")

    model = load_model(MODEL_SAVE_PATH)
    print(f"   Model   loaded  <- {MODEL_SAVE_PATH}")

    with open(SCALER_SAVE_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"   Scaler  loaded  <- {SCALER_SAVE_PATH}")

    with open(ENCODER_SAVE_PATH, 'rb') as f:
        le = pickle.load(f)
    print(f"   Encoder loaded  <- {ENCODER_SAVE_PATH}")

    return model, scaler, le


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data():
    """
    Reshape Time.csv from wide (24 hour columns) to long format.
    Each row = one site + location + hour observation.
    'Location' is kept so we can distinguish directions at the same site.

    Hour columns come from config.HOUR_COLUMNS, so this stays in sync
    with model_utils.load_long_data() used at inference time.
    """
    df = pd.read_csv(DATA_PATH)

    rows = []
    for _, row in df.iterrows():
        for hour_num, col in enumerate(HOUR_COLUMNS):
            rows.append({
                'scats_number':  row['SCATS Number'],
                'location':      row['Location'],
                'hour':          hour_num,
                'flow_per_hour': row[col]
            })

    return pd.DataFrame(rows)


# ── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(df):
    """
    - Encode 'location' string -> integer with LabelEncoder
    - Scale [scats_number, location_enc, hour, flow_per_hour] to [0,1]
    - Save both the scaler and the label encoder for later inference
    """
    le = LabelEncoder()
    df['location_enc'] = le.fit_transform(df['location'])

    os.makedirs('models', exist_ok=True)
    with open(ENCODER_SAVE_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f" Label encoder saved to {ENCODER_SAVE_PATH}")
    print(f" Known locations ({len(le.classes_)}):")
    for i, loc in enumerate(le.classes_):
        print(f"   [{i}] {loc}")

    features    = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data        = df[features].values
    scaler      = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved to {SCALER_SAVE_PATH}")

    return data_scaled, scaler, le


# ── 3. Create sequences ──────────────────────────────────────────────────────
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, -1])   # flow_per_hour is last column
    return np.array(X), np.array(y)


# ── 4. Build model ───────────────────────────────────────────────────────────
def build_rnn(input_shape):
    """
    Two-layer SimpleRNN with Dropout.
    Input has 4 features: scats_number, location_enc, hour, flow_per_hour.
    """
    model = Sequential([
        SimpleRNN(64, return_sequences=True,  input_shape=input_shape,
                  activation='tanh'),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mean_squared_error']
    )
    model.summary()
    return model


# ── 5. Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    n_features  = scaler.n_features_in_

    def inverse_flow(values):
        dummy = np.zeros((len(values), n_features))
        dummy[:, -1] = values.flatten()
        return scaler.inverse_transform(dummy)[:, -1]

    y_actual = inverse_flow(y_test)
    y_pred   = inverse_flow(predictions)

    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100

    print("\n RNN Evaluation Results:")
    print(f"   MAE:  {mae:.2f} vehicles/hour")
    print(f"   RMSE: {rmse:.2f} vehicles/hour")
    print(f"   MAPE: {mape:.2f}%")

    return mae, rmse, mape, y_actual, y_pred


# ── 6. Plot ──────────────────────────────────────────────────────────────────
def plot_results(y_actual, y_pred, title='RNN Predictions vs Actual'):
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual[:100], label='Actual',    color='blue')
    plt.plot(y_pred[:100],   label='Predicted', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Flow (vehicles/hour)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/rnn_predictions.png')
    plt.show()
    print(" Plot saved to models/rnn_predictions.png")


# ── 7. Train pipeline ────────────────────────────────────────────────────────
def train_and_save() -> tuple:
    """
    Full training pipeline. Returns (model, scaler, le, mae, rmse, mape).
    Called only when at least one artefact is missing.

    SEQ_LENGTH, EPOCHS, BATCH_SIZE, and TEST_SPLIT all come from
    config.py, so training stays in sync with inference (model_utils.py)
    and with the other two architectures (LSTM.py / GRU.py).
    """
    print("\n Starting full training pipeline ...")
    print(f" Using SEQ_LENGTH={SEQ_LENGTH}, EPOCHS={EPOCHS}, "
          f"BATCH_SIZE={BATCH_SIZE}, TEST_SPLIT={TEST_SPLIT} (from config.py)")

    df                       = load_data()
    data_scaled, scaler, le  = preprocess(df)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    print(f" Sequences created: X={X.shape}, y={y.shape}")

    split                    = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test          = X[:split], X[split:]
    y_train, y_test          = y[:split], y[split:]

    model                    = build_rnn(input_shape=(SEQ_LENGTH, X.shape[2]))
    early_stop               = EarlyStopping(monitor='val_loss', patience=5,
                                             restore_best_weights=True)

    print("\n Training RNN ...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n Model saved  -> {MODEL_SAVE_PATH}")

    mae, rmse, mape, y_actual, y_pred = evaluate(model, X_test, y_test, scaler)
    plot_results(y_actual, y_pred)

    return model, scaler, le, mae, rmse, mape


# ── 8. Main entry ────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RNN - Traffic Prediction")
    print("=" * 55)

    if all_artefacts_exist():
        # ── Fast path: every artefact is already on disk ──────────────────
        model, scaler, le = load_artefacts()
    else:
        # ── Slow path: train from scratch and save artefacts ──────────────
        model, scaler, le, mae, rmse, mape = train_and_save()
        print(f"\n Training complete  |  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    return model, scaler, le


# ── 9. Test a specific site + direction + hour ──────────────────────────────────
def test_prediction(site_id, location, hour, model, scaler, le):
    """
    Predict traffic flow for a specific site, direction, and hour.

    Parameters
    ----------
    site_id  : int    - SCATS Number            e.g. 2000
    location : str    - Direction string         e.g. 'WARRIGAL_RD N of TOORAK_RD'
    hour     : int    - Target hour (0-23)       e.g. 8
    model    : trained Keras model
    scaler   : fitted MinMaxScaler
    le       : fitted LabelEncoder for locations
    """
    if location not in le.classes_:
        print(f"\n Unknown location: '{location}'")
        print(" Available locations:")
        for loc in le.classes_:
            print(f"   {loc}")
        return

    location_enc = le.transform([location])[0]

    df        = load_data()
    site_data = df[
        (df['scats_number'] == site_id) &
        (df['location']     == location)
    ].copy()

    if len(site_data) < SEQ_LENGTH + 1:
        print(f" Not enough data for site {site_id} / {location}")
        return

    features                  = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    site_data['location_enc'] = location_enc
    data_scaled                = scaler.transform(site_data[features].values)

    for i in range(len(data_scaled) - SEQ_LENGTH):
        if site_data.iloc[i + SEQ_LENGTH]['hour'] == hour:
            sequence       = data_scaled[i : i + SEQ_LENGTH].reshape(1, SEQ_LENGTH, 4)
            pred_scaled    = model.predict(sequence, verbose=0)
            dummy          = np.zeros((1, 4))
            dummy[0, -1]   = pred_scaled[0][0]
            predicted_flow = scaler.inverse_transform(dummy)[0, -1]
            print(f"\n Site      : {site_id}")
            print(f" Direction : {location}")
            print(f" Hour      : {hour}:00")
            print(f" Predicted : {predicted_flow:.0f} vehicles/hour\n")
            return

    print(f" Hour {hour} not found for site {site_id} / {location}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, scaler, le = main()

    # ── Example predictions - always runs after model is ready ─────────────
    print("\n" + "=" * 55)
    print("  Running test predictions ...")
    print("=" * 55)

    test_prediction(2000, 'WARRIGAL_RD N of TOORAK_RD',   8,  model, scaler, le)
    test_prediction(2000, 'BURWOOD_HWY E of WARRIGAL_RD', 8,  model, scaler, le)
    test_prediction(2000, 'WARRIGAL_RD S of BURWOOD_HWY', 17, model, scaler, le)
    test_prediction(2000, 'TOORAK_RD W of WARRIGAL_RD',   3,  model, scaler, le)
    test_prediction(3002, 'DENMARK_ST N of BARKERS_RD',   17, model, scaler, le)
    test_prediction(2820, 'EARL_ST SE of PRINCESS_ST', 16, model, scaler, le)