"""
Traffic Prediction using SimpleRNN (TensorFlow/Keras)
======================================================
Dataset  : SCATS traffic data  (Time.csv)
Goal     : Predict the next hour's vehicle flow given the previous 24 hours,
           for a specific SCATS site AND direction (Location string).

Key change vs original:
  - 'Location' is now encoded and included as a feature alongside
    scats_number, hour, and flow_per_hour.
  - test_prediction() now accepts a `location` string so you can target
    a specific approach direction at a site.
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

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LENGTH       = 24
EPOCHS           = 50
BATCH_SIZE       = 32
TEST_SPLIT       = 0.2
MODEL_SAVE_PATH  = 'models/rnn_model.keras'
SCALER_SAVE_PATH = 'models/rnn_scaler.pkl'
ENCODER_SAVE_PATH = 'models/rnn_label_encoder.pkl'


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    """
    Reshape Time.csv from wide (24 hour columns) to long format.
    Each row = one site + location + hour observation.
    'Location' is kept so we can distinguish directions at the same site.
    """
    df = pd.read_csv('Dataset/Time.csv')

    hour_cols = [
        '12AM','1AM','2AM','3AM','4AM','5AM','6AM','7AM',
        '8AM','9AM','10AM','11AM','12PM','13PM','14PM','15PM',
        '16PM','17PM','18PM','19PM','20PM','21PM','22PM','23PM'
    ]

    rows = []
    for _, row in df.iterrows():
        for hour_num, col in enumerate(hour_cols):
            rows.append({
                'scats_number':  row['SCATS Number'],
                'location':      row['Location'],       # direction string
                'hour':          hour_num,
                'flow_per_hour': row[col]
            })

    return pd.DataFrame(rows)


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df):
    """
    - Encode 'location' string → integer with LabelEncoder
    - Scale [scats_number, location_enc, hour, flow_per_hour] to [0,1]
    - Save both the scaler and the label encoder for later inference
    """
    # Encode location strings to integers
    le = LabelEncoder()
    df['location_enc'] = le.fit_transform(df['location'])

    os.makedirs('models', exist_ok=True)
    with open(ENCODER_SAVE_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f" Label encoder saved to {ENCODER_SAVE_PATH}")
    print(f" Known locations ({len(le.classes_)}):")
    for i, loc in enumerate(le.classes_):
        print(f"   [{i}] {loc}")

    features = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved to {SCALER_SAVE_PATH}")

    return data_scaled, scaler, le


# ── 3. Create sequences ───────────────────────────────────────────────────────
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, -1])   # flow_per_hour is last column
    return np.array(X), np.array(y)


# ── 4. Build model ────────────────────────────────────────────────────────────
def build_rnn(input_shape):
    """
    Two-layer SimpleRNN with Dropout, matching GRU.py's capacity.
    Input now has 4 features: scats_number, location_enc, hour, flow_per_hour
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


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
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


# ── 6. Plot ───────────────────────────────────────────────────────────────────
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


# ── 7. Train ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print(" RNN Training")
    print("=" * 50)

    df = load_data()
    data_scaled, scaler, le = preprocess(df)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    print(f" Sequences created: X={X.shape}, y={y.shape}")

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_rnn(input_shape=(SEQ_LENGTH, X.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)

    print("\n Training RNN...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n Model saved to {MODEL_SAVE_PATH}")

    mae, rmse, mape, y_actual, y_pred = evaluate(model, X_test, y_test, scaler)
    plot_results(y_actual, y_pred)

    return mae, rmse, mape


# ── 8. Test a specific site + direction + hour ────────────────────────────────
def test_prediction(site_id, location, hour, model, scaler, le):
    """
    Predict traffic flow for a specific site, direction, and hour.

    Parameters
    ----------
    site_id  : int    – SCATS Number            e.g. 2000
    location : str    – Direction string         e.g. 'WARRIGAL_RD N of TOORAK_RD'
    hour     : int    – Target hour (0–23)       e.g. 8
    model    : trained Keras model
    scaler   : fitted MinMaxScaler
    le       : fitted LabelEncoder for locations

    Available directions for site 2000:
      'WARRIGAL_RD N of TOORAK_RD'
      'BURWOOD_HWY E of WARRIGAL_RD'
      'WARRIGAL_RD S of BURWOOD_HWY'
      'TOORAK_RD W of WARRIGAL_RD'
    """
    # Validate location string
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

    features    = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    site_data['location_enc'] = location_enc
    data_scaled = scaler.transform(site_data[features].values)

    for i in range(len(data_scaled) - SEQ_LENGTH):
        if site_data.iloc[i + SEQ_LENGTH]['hour'] == hour:
            sequence       = data_scaled[i : i + SEQ_LENGTH].reshape(1, SEQ_LENGTH, 4)
            pred_scaled    = model.predict(sequence, verbose=0)
            dummy          = np.zeros((1, 4))
            dummy[0, -1]   = pred_scaled[0][0]
            predicted_flow = scaler.inverse_transform(dummy)[0, -1]
            print(f"\n Site    : {site_id}")
            print(f" Direction: {location}")
            print(f" Hour     : {hour}:00")
            print(f" Predicted Flow: {predicted_flow:.0f} vehicles/hour\n")
            return

    print(f" Hour {hour} not found for site {site_id} / {location}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()

    model = load_model(MODEL_SAVE_PATH)
    with open(SCALER_SAVE_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_SAVE_PATH, 'rb') as f:
        le = pickle.load(f)

    # Example predictions — site + direction + hour
    test_prediction(2000, 'WARRIGAL_RD N of TOORAK_RD',   8,  model, scaler, le)
    test_prediction(2000, 'BURWOOD_HWY E of WARRIGAL_RD', 8,  model, scaler, le)
    test_prediction(2000, 'WARRIGAL_RD S of BURWOOD_HWY', 17, model, scaler, le)
    test_prediction(2000, 'TOORAK_RD W of WARRIGAL_RD',   3,  model, scaler, le)
    test_prediction(3002, 'DENMARK_ST N of BARKERS_RD',   17, model, scaler, le)