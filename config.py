"""
config.py
=========
Central configuration shared by every script in the project
(predict_all / map renderer AND the route-finder CLI).

Keeping this data in one place means the map script and the
route-finder script can never drift out of sync with each other.
"""

# ── Sequence length used by every RNN-family model (RNN / LSTM / GRU) ──────────
# Must match SEQ_LENGTH in RNN.py / LSTM.py / GRU.py — the models were
# TRAINED expecting this many timesteps as input. If this value doesn't
# match the training scripts, model.predict() at inference time will either
# throw a shape-mismatch error or silently produce meaningless predictions.
SEQ_LENGTH = 3

# ── Dataset used to build the 24-hour sequences for prediction ─────────────────
DATA_PATH = 'Dataset/Time.csv'

# ── Shared training hyperparameters for RNN / LSTM / GRU ───────────────────────
# Centralised here so RNN.py / LSTM.py / GRU.py all train under identical
# conditions, making their MAE/RMSE/MAPE results directly comparable.
EPOCHS     = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.2

# ── Per-model artefact paths ────────────────────────────────────────────────────
# Add new entries here if you train additional architectures later
# (e.g. 'GRU', 'LSTM'). The naming pattern matches the existing
# rnn_model.keras / rnn_scaler.pkl / rnn_label_encoder.pkl files.
MODEL_CONFIGS = {
    'RNN': {
        'model_path':   'models/rnn_model.keras',
        'scaler_path':  'models/rnn_scaler.pkl',
        'encoder_path': 'models/rnn_label_encoder.pkl',
    },
    'LSTM': {
        'model_path':   'models/lstm_model.keras',
        'scaler_path':  'models/lstm_scaler.pkl',
        'encoder_path': 'models/lstm_label_encoder.pkl',
    },
    'GRU': {
        'model_path':   'models/gru_model.keras',
        'scaler_path':  'models/gru_scaler.pkl',
        'encoder_path': 'models/gru_label_encoder.pkl',
    },
}

# ── Site coordinates (lat, lon) ─────────────────────────────────────────────────
COORDS = {
    2000: (-37.851923, 145.094324),
    2820: (-37.794785, 145.030465),
    2825: (-37.786610, 145.062020),
    2827: (-37.781739, 145.077331),
    3002: (-37.815162, 145.026572),
    3120: (-37.822895, 145.057288),
    3127: (-37.825227, 145.077947),
    3180: (-37.796213, 145.083507),
    3662: (-37.808953, 145.027457),
    3682: (-37.837475, 145.096868),
    4032: (-37.802297, 145.061230),
    4043: (-37.847160, 145.052627),
    4051: (-37.794143, 145.069333),
    4057: (-37.804969, 145.081760),
    4263: (-37.823056, 145.024958),
    4266: (-37.825268, 145.043338),
    4270: (-37.830228, 145.032815),
    4321: (-37.800838, 145.049119),
}

# ── Road connections: (site_a, site_b, distance_km) ─────────────────────────────
CONNECTIONS = [
    (2827, 4051, 1.85),
    (4051, 3180, 1.57),
    (3180, 4057, 1.07),
    (4057, 4032, 2.47),
    (4057, 3127, 2.53),
    (3127, 3682, 3.57),
    (3120, 3127, 2.62),
    (2000, 3682, 1.75),
    (2000, 4043, 4.21),
    (4043, 3120, 2.86),
    (3120, 4032, 2.45),
    (3120, 4266, 1.87),
    (4043, 4270, 3.55),
    (4266, 4270, 1.67),
    (4270, 4263, 1.49),
    (4266, 4263, 2.14),
    (4266, 3002, 2.72),
    (4263, 3002, 1.15),
    (3002, 3662, 0.99),
    (3662, 4321, 2.38),
    (3662, 2820, 1.69),
    (2820, 4321, 3.54),
    (4321, 4032, 1.64),
    (4032, 2825, 1.96),
]

# ── Human-readable site names ───────────────────────────────────────────────────
SITE_NAMES = {
    2000: "WARRIGAL_RD / TOORAK_RD",
    2820: "TOORAK_RD / KOOYONG_RD",
    2825: "TOORAK_RD / BURKE_RD",
    2827: "TOORAK_RD / GLENFERRIE_RD",
    3002: "DENMARK_ST / BARKERS_RD",
    3120: "BURKE_RD / COTHAM_RD",
    3127: "CAMBERWELL_RD / RIVERSDALE_RD",
    3180: "BURKE_RD / BARKERS_RD",
    3662: "HIGH_ST / BURKE_RD",
    3682: "BURKE_RD / HIGH_ST",
    4032: "WHITEHORSE_RD / RATHMINES_RD",
    4043: "CANTERBURY_RD / MONT_ALBERT_RD",
    4051: "WHITEHORSE_RD / SPRINGVALE_RD",
    4057: "CANTERBURY_RD / BALWYN_RD",
    4263: "DONCASTER_RD / ELGAR_RD",
    4266: "AUBURN_RD / BURWOOD_RD",
    4270: "BURWOOD_RD / HIGHFIELD_RD",
    4321: "TOORAK_RD / AUBURN_RD",
}

# ── Hour-of-day labels (index 0 = 12AM ... index 23 = 11PM) ─────────────────────
HOUR_LABELS = [
    "12AM", "1AM", "2AM", "3AM", "4AM", "5AM",
    "6AM", "7AM", "8AM", "9AM", "10AM", "11AM",
    "12PM", "1PM", "2PM", "3PM", "4PM", "5PM",
    "6PM", "7PM", "8PM", "9PM", "10PM", "11PM",
]

# ── Hour column names as they appear in Dataset/Time.csv ────────────────────────
HOUR_COLUMNS = [
    '12AM', '1AM', '2AM', '3AM', '4AM', '5AM',
    '6AM', '7AM', '8AM', '9AM', '10AM', '11AM',
    '12PM', '13PM', '14PM', '15PM', '16PM', '17PM',
    '18PM', '19PM', '20PM', '21PM', '22PM', '23PM',
]