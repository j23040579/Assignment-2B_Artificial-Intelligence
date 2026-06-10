#loading libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import os

# uses the last 24 hours as input to predict the next hour
SEQ_LENGTH = 24
EPOCHS = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.2
MODEL_SAVE_PATH = 'models/lstm_model.keras'
SCALER_SAVE_PATH = 'models/lstm_scaler.pkl'
ENCODER_SAVE_PATH = 'models/lstm_label_encoder.pkl'


def load_data():
    df = pd.read_csv('Dataset/Time.csv')
    
    # Hour columns and their numeric values
    hour_cols = ['12AM','1AM','2AM','3AM','4AM','5AM','6AM','7AM',
                 '8AM','9AM','10AM','11AM','12PM','13PM','14PM','15PM',
                 '16PM','17PM','18PM','19PM','20PM','21PM','22PM','23PM']
    
    # Convert wide format to long format
    rows = []
    for _, row in df.iterrows():
        for hour_num, col in enumerate(hour_cols):
            rows.append({
                'scats_number':  row['SCATS Number'],
                'location':      row['Location'],
                'hour':          hour_num,
                'flow_per_hour': row[col]
            })
    
    result = pd.DataFrame(rows)
    return result


# Normalise features and create sequences for LSTM input
def preprocess(df):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['location_enc'] = le.fit_transform(df['location'])

    os.makedirs('models', exist_ok=True)
    with open(ENCODER_SAVE_PATH, 'wb') as f:
        pickle.dump(le, f)

    features = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved to {SCALER_SAVE_PATH}")

    return data_scaled, scaler, le



def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, -1])
    return np.array(X), np.array(y)


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
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


def evaluate(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    n_features = scaler.n_features_in_

    def inverse_flow(values):
        dummy = np.zeros((len(values), n_features))
        dummy[:, -1] = values.flatten()
        return scaler.inverse_transform(dummy)[:, -1]

    y_actual = inverse_flow(y_test)
    y_pred = inverse_flow(predictions)

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100

    # model studies the data and learns the patterns
    # MAE,RMSE and MAPE measures how far off the pattern was
    print("\n LSTM Evaluation Results:")
    print(f"   MAE:  {mae:.2f} vehicles/hour") # Actual
    print(f"   RMSE: {rmse:.2f} vehicles/hour") # Predicted
    print(f"   MAPE: {mape:.2f}%") # average error as a percentage of the actual value

    return mae, rmse, mape, y_actual, y_pred


# showing a graph of what it predicted vs what it actually got based on the dataset
def plot_results(y_actual, y_pred, title='LSTM Predictions vs Actual'):
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual[:100], label='Actual', color='blue')
    plt.plot(y_pred[:100], label='Predicted', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Flow (vehicles/hour)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# trains the model base on the time.csv dataset
def main():
    print("=" * 50)
    print(" LSTM Training ")
    print("=" * 50)

    df = load_data()
    data_scaled, scaler, le = preprocess(df)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    print(f" Sequences created: X={X.shape}, y={y.shape}")

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(input_shape=(SEQ_LENGTH, X.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\n Training LSTM...")
    history = model.fit(
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

# testing specific route

def test_prediction(site_id, location, hour, model, scaler, le):
    # Load data and find real sequences for this site and hour
    df = load_data()
    
    # Filter for specific site
    site_data = df[
        (df['scats_number'] == site_id) &
        (df['location'] == location)
    ].copy()
    
    if len(site_data) < SEQ_LENGTH + 1:
        print(f"Not enough data for site {site_id}")
        return
    
    # Normalise
    location_enc = le.transform([location])[0]
    site_data ['location_enc'] = location_enc
    features = ['scats_number', 'location_enc', 'hour', 'flow_per_hour']
    data_scaled = scaler.transform(site_data[features].values)
    
    # Find a sequence ending at the target hour
    for i in range(len(data_scaled) - SEQ_LENGTH):
        if site_data.iloc[i + SEQ_LENGTH]['hour'] == hour:
            sequence = data_scaled[i:i + SEQ_LENGTH].reshape(1, SEQ_LENGTH, 4)
            prediction_scaled = model.predict(sequence, verbose=0)
            dummy = np.zeros((1, 4))
            dummy[0, -1] = prediction_scaled[0][0]
            predicted_flow = scaler.inverse_transform(dummy)[0, -1]
            print(f"\nSite {site_id} at {hour}:00")
            print(f"Location: {location}")
            print(f"Predicted Flow: {predicted_flow:.0f} vehicles/hour\n")
            return
    
    print(f"Hour {hour} not found for site {site_id}")

    
if __name__ == '__main__':
    main()

    model = load_model(MODEL_SAVE_PATH)
    with open(SCALER_SAVE_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_SAVE_PATH, 'rb') as f:
        le = pickle.load(f)

    print("\n" + "=" * 55)
    print("  Running test predictions ...")
    print("=" * 55)

    test_prediction(2000, 'WARRIGAL_RD N of TOORAK_RD',   8,  model, scaler, le) # morning
    test_prediction(2000, 'BURWOOD_HWY E of WARRIGAL_RD', 8,  model, scaler, le) # morning
    test_prediction(2000, 'WARRIGAL_RD S of BURWOOD_HWY', 17, model, scaler, le) #evening
    test_prediction(2000, 'TOORAK_RD W of WARRIGAL_RD',   3,  model, scaler, le) #midnight
    test_prediction(3002, 'DENMARK_ST N of BARKERS_RD',   17, model, scaler, le) #evening
    test_prediction(2820, 'EARL_ST SE of PRINCESS_ST', 16, model, scaler, le) #evening


    