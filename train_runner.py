"""
train_runner.py
===============
Centralised training dispatcher for A2B route prediction.

When a model artefact (model, scaler, or label-encoder) is missing from
disk, A2B.py delegates here.  This module imports each model's own
training pipeline and calls its main() so that the exact same
hyper-parameters, paths, and pre-processing logic are always used.

Supported model keys (case-insensitive):
    'RNN'  -> trains models/rnn_model.keras  + pkl files via RNN.py
    'LSTM' -> trains models/lstm_model.keras + pkl files via LSTM.py
    'GRU'  -> trains models/gru_model.keras  + pkl files via GRU.py

Public API
----------
    train_model(model_name: str) -> None
        Train the requested model and save all artefacts to disk.
        Raises ValueError for unrecognised model names.
        Raises RuntimeError if the training pipeline raises an exception.
"""

import importlib
import os
import sys
import time


# ── Registry: maps each model key -> (module_file, display_name) ────────────
# The module_file is the importable module name (without .py).
# Keep module names lower-case so they match the actual filenames.
_MODEL_REGISTRY: dict[str, dict] = {
    "RNN": {
        "module":   "RNN",
        "display":  "SimpleRNN",
        "artefacts": [
            "models/rnn_model.keras",
            "models/rnn_scaler.pkl",
            "models/rnn_label_encoder.pkl",
        ],
    },
    "LSTM": {
        "module":   "LSTM",
        "display":  "LSTM",
        "artefacts": [
            "models/lstm_model.keras",
            "models/lstm_scaler.pkl",
            "models/lstm_label_encoder.pkl",
        ],
    },
    "GRU": {
        "module":   "GRU",
        "display":  "GRU",
        "artefacts": [
            "models/gru_model.keras",
            "models/gru_scaler.pkl",
            "models/gru_label_encoder.pkl",
        ],
    },
}


def _missing_artefacts(model_key: str) -> list[str]:
    """
    Return a list of artefact paths that do not yet exist on disk
    for the given model key.  Returns an empty list if all are present.
    """
    entry = _MODEL_REGISTRY[model_key]
    return [p for p in entry["artefacts"] if not os.path.isfile(p)]


def artefacts_exist(model_name: str) -> bool:
    """
    Return True if every saved artefact for *model_name* is already on disk.

    Parameters
    ----------
    model_name : str
        Case-insensitive model key: 'RNN', 'LSTM', or 'GRU'.

    Raises
    ------
    ValueError
        If *model_name* is not in the registry.
    """
    key = model_name.upper()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return len(_missing_artefacts(key)) == 0


def train_model(model_name: str) -> None:
    """
    Import the appropriate training module and run its main() pipeline.

    The function prints progress banners so the user can see what is
    happening.  After training, it verifies that all expected artefacts
    were actually written to disk.

    Parameters
    ----------
    model_name : str
        Case-insensitive model key: 'RNN', 'LSTM', or 'GRU'.

    Raises
    ------
    ValueError
        If *model_name* is not recognised.
    RuntimeError
        If the training pipeline fails or the expected artefacts are
        still missing after training completes.
    """
    key = model_name.upper()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {sorted(_MODEL_REGISTRY.keys())}"
        )

    entry       = _MODEL_REGISTRY[key]
    module_name = entry["module"]
    display     = entry["display"]
    missing     = _missing_artefacts(key)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Auto-training {display} model — artefacts not found on disk")
    print("=" * 60)
    print(f"  Missing file(s):")
    for p in missing:
        print(f"    ✗  {p}")
    print()

    # ── Ensure the project root is on sys.path so RNN/LSTM/GRU are importable ─
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Dynamically import the training module ─────────────────────────────────
    try:
        training_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import training module '{module_name}.py'. "
            f"Make sure the file exists in the project root.\n"
            f"Original error: {exc}"
        ) from exc

    # ── Run the training pipeline ──────────────────────────────────────────────
    t_start = time.time()
    try:
        # Every training module exposes a main() that trains and saves artefacts.
        # RNN.main() also returns (model, scaler, le); LSTM/GRU return metrics.
        # We don't need the return value here — we just need the side effects.
        training_module.main()
    except Exception as exc:
        raise RuntimeError(
            f"Training pipeline for '{key}' raised an exception:\n{exc}"
        ) from exc

    elapsed = time.time() - t_start

    # ── Post-training verification ─────────────────────────────────────────────
    still_missing = _missing_artefacts(key)
    if still_missing:
        raise RuntimeError(
            f"Training completed but the following artefacts are still "
            f"missing — check the training script for errors:\n"
            + "\n".join(f"  ✗  {p}" for p in still_missing)
        )

    print()
    print("=" * 60)
    print(f"  {display} training complete in {elapsed:.1f}s")
    print("  Artefacts saved:")
    for p in entry["artefacts"]:
        size_kb = os.path.getsize(p) / 1024
        print(f"    ✔  {p}  ({size_kb:.1f} KB)")
    print("=" * 60 + "\n")
