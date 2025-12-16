#!/usr/bin/env python3
"""
Surrogate Model for PPL Prediction

Trains a simple model to predict perplexity from architecture features.
This allows screening 100s of candidate architectures before GPU training.

Usage:
    python scripts/surrogate_model.py           # Train and evaluate
    python scripts/surrogate_model.py --predict # Predict on new architectures
"""
import sys
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "arcfusion"))
from db import ArcFusionDB


@dataclass
class ArchFeatures:
    """Features extracted from architecture for PPL prediction."""
    n_layers: int
    n_kv_heads: int  # 1=MQA, 2=GQA, 4=GQA4, 8=MHA
    has_mamba: bool
    has_linear_attn: bool
    is_hybrid: bool
    is_fast_mamba: bool  # parallel scan
    d_model: int = 256
    n_heads: int = 8

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model."""
        return np.array([
            self.n_layers,
            self.n_kv_heads,
            float(self.has_mamba),
            float(self.has_linear_attn),
            float(self.is_hybrid),
            float(self.is_fast_mamba),
            self.d_model,
            self.n_heads,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            'n_layers', 'n_kv_heads', 'has_mamba', 'has_linear_attn',
            'is_hybrid', 'is_fast_mamba', 'd_model', 'n_heads'
        ]


def extract_features(model_name: str, n_layers_db: int = 4, d_model: int = 256) -> ArchFeatures:
    """Extract features from model name."""
    name = model_name.lower()

    # Determine layer count from name or DB
    # Handle patterns like GQA410 (GQA4 with 10 layers), MHA32, DeepGQA14
    n_layers = n_layers_db

    # Try specific patterns first
    if 'gqa4' in name:
        # GQA4 with layers: GQA410 = 10 layers, GQA418 = 18 layers
        match = re.search(r'gqa4(\d+)', name)
        if match:
            n_layers = int(match.group(1))
    elif 'deepgqa' in name:
        match = re.search(r'deepgqa(\d+)', name)
        if match:
            n_layers = int(match.group(1))
    elif re.search(r'(mha|gqa|mqa)(\d+)$', name):
        # MHA32, GQA14, MQA18
        match = re.search(r'(mha|gqa|mqa)(\d+)$', name)
        if match:
            n_layers = int(match.group(2))
    elif re.search(r'(\d+)$', name):
        # Fallback: number at end
        match = re.search(r'(\d+)$', name)
        if match:
            n_layers = int(match.group(1))

    # Determine KV heads from attention type
    if 'mqa' in name:
        n_kv_heads = 1
    elif 'gqa4' in name:
        n_kv_heads = 4
    elif 'gqa' in name or 'deepgqa' in name:
        n_kv_heads = 2
    elif 'mha' in name:
        n_kv_heads = 8
    elif 'mamba' in name:
        n_kv_heads = 0  # No attention
    elif 'linear' in name:
        n_kv_heads = 8  # Linear attention uses all heads
    else:
        n_kv_heads = 8  # Default to MHA

    # Architecture flags
    has_mamba = 'mamba' in name
    has_linear_attn = 'linear' in name
    is_hybrid = any(x in name for x in ['hybrid', 'sandwich', 'mix', '4to1', '5to1', 'heavy'])
    is_fast_mamba = 'mambafast' in name or 'fastmamba' in name or name.endswith('fast')

    return ArchFeatures(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        has_mamba=has_mamba,
        has_linear_attn=has_linear_attn,
        is_hybrid=is_hybrid,
        is_fast_mamba=is_fast_mamba,
        d_model=d_model,
        n_heads=8,
    )


def load_training_data(db: ArcFusionDB) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load features and targets from training_runs table.

    Returns: (X, y_ppl, y_time, names)
    """
    runs = db.conn.execute('''
        SELECT model_name, n_layers, d_model, perplexity, time_seconds
        FROM training_runs
        WHERE success = 1 AND perplexity IS NOT NULL
        ORDER BY created_at
    ''').fetchall()

    X = []
    y_ppl = []
    y_time = []
    names = []

    for model_name, n_layers_db, d_model, ppl, time_s in runs:
        # Skip runs with missing time data - they would skew the time model
        if time_s is None or time_s <= 0:
            continue
        features = extract_features(model_name, n_layers_db or 4, d_model or 256)
        X.append(features.to_vector())
        y_ppl.append(ppl)
        y_time.append(time_s)
        names.append(model_name)

    return np.array(X), np.array(y_ppl), np.array(y_time), names


class SurrogateModel:
    """Surrogate model for PPL and training time prediction."""

    def __init__(self):
        # PPL prediction params
        self.weights_ppl = None
        self.mean_y_ppl = None
        self.std_y_ppl = None
        # Time prediction params
        self.weights_time = None
        self.mean_y_time = None
        self.std_y_time = None
        # Shared feature normalization
        self.mean_x = None
        self.std_x = None
        # Training metadata
        self.n_training_samples = 0
        # Legacy compatibility
        self.weights = None
        self.bias = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X: np.ndarray, y_ppl: np.ndarray, y_time: np.ndarray = None):
        """Fit linear regression for PPL and optionally time."""
        # Track training data size
        self.n_training_samples = len(X)

        # Standardize features
        self.mean_x = X.mean(axis=0)
        self.std_x = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_x) / self.std_x
        X_bias = np.column_stack([np.ones(len(X)), X_norm])

        # L2 regularization matrix
        lambda_reg = 0.1
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0  # Don't regularize bias
        XtX_reg = X_bias.T @ X_bias + lambda_reg * I

        # Fit PPL model
        self.mean_y_ppl = y_ppl.mean()
        self.std_y_ppl = y_ppl.std() + 1e-8
        y_ppl_norm = (y_ppl - self.mean_y_ppl) / self.std_y_ppl
        self.weights_ppl = np.linalg.solve(XtX_reg, X_bias.T @ y_ppl_norm)

        # Legacy compatibility
        self.weights = self.weights_ppl
        self.mean_y = self.mean_y_ppl
        self.std_y = self.std_y_ppl

        # Fit time model if provided
        if y_time is not None:
            self.mean_y_time = y_time.mean()
            self.std_y_time = y_time.std() + 1e-8
            y_time_norm = (y_time - self.mean_y_time) / self.std_y_time
            self.weights_time = np.linalg.solve(XtX_reg, X_bias.T @ y_time_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict PPL from features (legacy compatibility)."""
        return self.predict_ppl(X)

    def predict_ppl(self, X: np.ndarray) -> np.ndarray:
        """Predict PPL from features."""
        X_norm = (X - self.mean_x) / self.std_x
        X_bias = np.column_stack([np.ones(len(X)), X_norm])
        y_norm = X_bias @ self.weights_ppl
        return y_norm * self.std_y_ppl + self.mean_y_ppl

    def predict_time(self, X: np.ndarray) -> np.ndarray:
        """Predict training time from features."""
        if self.weights_time is None:
            raise ValueError("Time model not fitted. Call fit() with y_time parameter.")
        X_norm = (X - self.mean_x) / self.std_x
        X_bias = np.column_stack([np.ones(len(X)), X_norm])
        y_norm = X_bias @ self.weights_time
        return y_norm * self.std_y_time + self.mean_y_time

    def predict_both(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict both PPL and time. Returns (ppl, time)."""
        ppl = self.predict_ppl(X)
        time = self.predict_time(X) if self.weights_time is not None else np.zeros(len(X))
        return ppl, time

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'weights_ppl': self.weights_ppl,
                'weights_time': self.weights_time,
                'mean_x': self.mean_x,
                'std_x': self.std_x,
                'mean_y_ppl': self.mean_y_ppl,
                'std_y_ppl': self.std_y_ppl,
                'mean_y_time': self.mean_y_time,
                'std_y_time': self.std_y_time,
                'n_training_samples': self.n_training_samples,
                # Legacy fields
                'weights': self.weights_ppl,
                'bias': None,
                'mean_y': self.mean_y_ppl,
                'std_y': self.std_y_ppl,
            }, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Try new format first
            if 'weights_ppl' in data:
                self.weights_ppl = data['weights_ppl']
                self.weights_time = data.get('weights_time')
                self.mean_y_ppl = data['mean_y_ppl']
                self.std_y_ppl = data['std_y_ppl']
                self.mean_y_time = data.get('mean_y_time')
                self.std_y_time = data.get('std_y_time')
            else:
                # Legacy format
                self.weights_ppl = data['weights']
                self.mean_y_ppl = data['mean_y']
                self.std_y_ppl = data['std_y']
            self.mean_x = data['mean_x']
            self.std_x = data['std_x']
            self.n_training_samples = data.get('n_training_samples', 0)
            # Legacy compatibility
            self.weights = self.weights_ppl
            self.mean_y = self.mean_y_ppl
            self.std_y = self.std_y_ppl


def retrain_if_needed(db: ArcFusionDB, model_path: str, min_new_samples: int = 3) -> tuple[bool, str]:
    """Retrain surrogate model if enough new training data exists.

    Args:
        db: Database connection
        model_path: Path to surrogate model file
        min_new_samples: Minimum new samples required to trigger retrain

    Returns:
        (retrained, message) tuple
    """
    # Load current model to check training sample count
    model = SurrogateModel()
    if Path(model_path).exists():
        model.load(model_path)
        old_samples = model.n_training_samples
    else:
        old_samples = 0

    # Check current data count
    X, y_ppl, y_time, names = load_training_data(db)
    current_samples = len(X)
    new_samples = current_samples - old_samples

    if new_samples < min_new_samples:
        return False, f"Only {new_samples} new samples (need {min_new_samples})"

    # Retrain on all data
    model.fit(X, y_ppl, y_time)
    model.save(model_path)

    # Quick evaluation on training data
    ppl_pred = model.predict_ppl(X)
    mae = np.mean(np.abs(ppl_pred - y_ppl))
    corr = np.corrcoef(ppl_pred, y_ppl)[0, 1]

    # Update predictions for untrained dream candidates
    n_updated = update_untrained_candidate_predictions(db, model)

    # Check prediction accuracy on dream candidates (predicted vs actual)
    accuracy = db.get_surrogate_accuracy_stats()
    accuracy_msg = ""
    if accuracy.get('n_samples', 0) >= 2 and not accuracy.get('insufficient_data'):
        accuracy_msg = f" Prediction accuracy: MAPE={accuracy['ppl_mape']:.1f}%"

    base_msg = f"Retrained on {current_samples} samples (+{new_samples}). MAE={mae:.1f}, corr={corr:.3f}"
    if n_updated > 0:
        return True, f"{base_msg}. Updated {n_updated} candidate predictions.{accuracy_msg}"

    return True, f"{base_msg}{accuracy_msg}"


def update_untrained_candidate_predictions(db: ArcFusionDB, model: SurrogateModel) -> int:
    """Update surrogate predictions for all untrained dream candidates.

    After retraining the surrogate model on new data, this updates the
    predicted_ppl and predicted_time for candidates that haven't been
    trained yet, giving them more accurate estimates.

    Returns: Number of candidates updated
    """
    # Get all untrained candidates (no practical limit)
    candidates = db.list_dream_candidates(untrained_only=True, limit=100000)
    if not candidates:
        return 0

    updated = 0
    for cand in candidates:
        # Build feature vector from candidate attributes
        features = ArchFeatures(
            n_layers=cand.n_layers,
            n_kv_heads=cand.n_kv_heads,
            has_mamba=cand.has_mamba,
            has_linear_attn=cand.has_linear_attn,
            is_hybrid=cand.is_hybrid,
            is_fast_mamba=cand.has_mamba,  # Assume parallel scan for Mamba
            d_model=256,  # Default from training config
            n_heads=8,
        )

        # Predict new PPL and time
        X = features.to_vector().reshape(1, -1)
        try:
            pred_ppl = float(model.predict_ppl(X)[0])
            pred_time = float(model.predict_time(X)[0]) if model.weights_time is not None else 0.0

            # Update if predictions changed significantly (> 1% difference)
            ppl_diff = abs(pred_ppl - cand.predicted_ppl) / max(cand.predicted_ppl, 1.0)
            time_diff = abs(pred_time - cand.predicted_time) / max(cand.predicted_time, 1.0) if cand.predicted_time > 0 else 1.0

            if ppl_diff > 0.01 or time_diff > 0.01:
                db.update_dream_candidate_predictions(cand.candidate_id, pred_ppl, pred_time)
                updated += 1
        except (ValueError, IndexError, TypeError):
            # Skip candidates with invalid features (malformed data)
            continue

    return updated


def evaluate_model(model: SurrogateModel, X: np.ndarray, y_ppl: np.ndarray,
                   y_time: np.ndarray, names: list[str], split_ratio: float = 0.8):
    """Evaluate model with train/test split for both PPL and time."""
    n = len(X)
    n_train = int(n * split_ratio)

    # Use chronological split (older data for training)
    X_train, X_test = X[:n_train], X[n_train:]
    y_ppl_train, y_ppl_test = y_ppl[:n_train], y_ppl[n_train:]
    y_time_train, y_time_test = y_time[:n_train], y_time[n_train:]
    names_test = names[n_train:]

    model.fit(X_train, y_ppl_train, y_time_train)

    # PPL metrics
    y_ppl_train_pred = model.predict_ppl(X_train)
    ppl_train_mae = np.abs(y_ppl_train - y_ppl_train_pred).mean()
    ppl_train_rmse = np.sqrt(((y_ppl_train - y_ppl_train_pred) ** 2).mean())

    y_ppl_test_pred = model.predict_ppl(X_test)
    ppl_test_mae = np.abs(y_ppl_test - y_ppl_test_pred).mean()
    ppl_test_rmse = np.sqrt(((y_ppl_test - y_ppl_test_pred) ** 2).mean())
    ppl_corr = np.corrcoef(y_ppl_test, y_ppl_test_pred)[0, 1] if len(y_ppl_test) > 1 else 0

    # Time metrics
    y_time_train_pred = model.predict_time(X_train)
    time_train_mae = np.abs(y_time_train - y_time_train_pred).mean()
    time_train_rmse = np.sqrt(((y_time_train - y_time_train_pred) ** 2).mean())

    y_time_test_pred = model.predict_time(X_test)
    time_test_mae = np.abs(y_time_test - y_time_test_pred).mean()
    time_test_rmse = np.sqrt(((y_time_test - y_time_test_pred) ** 2).mean())
    time_corr = np.corrcoef(y_time_test, y_time_test_pred)[0, 1] if len(y_time_test) > 1 else 0

    return {
        # PPL metrics
        'ppl_train_mae': ppl_train_mae,
        'ppl_train_rmse': ppl_train_rmse,
        'ppl_test_mae': ppl_test_mae,
        'ppl_test_rmse': ppl_test_rmse,
        'ppl_correlation': ppl_corr,
        # Time metrics
        'time_train_mae': time_train_mae,
        'time_train_rmse': time_train_rmse,
        'time_test_mae': time_test_mae,
        'time_test_rmse': time_test_rmse,
        'time_correlation': time_corr,
        # General
        'n_train': n_train,
        'n_test': n - n_train,
        'ppl_predictions': list(zip(names_test, y_ppl_test, y_ppl_test_pred)),
        'time_predictions': list(zip(names_test, y_time_test, y_time_test_pred)),
    }


def rank_candidates(model: SurrogateModel, candidates: list[ArchFeatures]) -> list[tuple[ArchFeatures, float]]:
    """Rank candidate architectures by predicted PPL."""
    X = np.array([c.to_vector() for c in candidates])
    preds = model.predict(X)
    ranked = sorted(zip(candidates, preds), key=lambda x: x[1])
    return ranked


def main():
    print("=" * 60)
    print("SURROGATE MODEL FOR PPL AND TIME PREDICTION")
    print("=" * 60)

    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Load data
    print("\nLoading training data...")
    X, y_ppl, y_time, names = load_training_data(db)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {ArchFeatures.feature_names()}")
    print(f"  PPL range: {y_ppl.min():.1f} - {y_ppl.max():.1f}")
    print(f"  Time range: {y_time.min():.1f}s - {y_time.max():.1f}s")

    # Train and evaluate
    print("\nTraining surrogate model (dual: PPL + Time)...")
    model = SurrogateModel()
    results = evaluate_model(model, X, y_ppl, y_time, names, split_ratio=0.8)

    print(f"\nPPL Prediction Results:")
    print(f"  Train MAE: {results['ppl_train_mae']:.1f} PPL")
    print(f"  Train RMSE: {results['ppl_train_rmse']:.1f} PPL")
    print(f"  Test MAE: {results['ppl_test_mae']:.1f} PPL")
    print(f"  Test RMSE: {results['ppl_test_rmse']:.1f} PPL")
    print(f"  Correlation: {results['ppl_correlation']:.3f}")

    print(f"\nTime Prediction Results:")
    print(f"  Train MAE: {results['time_train_mae']:.1f}s")
    print(f"  Train RMSE: {results['time_train_rmse']:.1f}s")
    print(f"  Test MAE: {results['time_test_mae']:.1f}s")
    print(f"  Test RMSE: {results['time_test_rmse']:.1f}s")
    print(f"  Correlation: {results['time_correlation']:.3f}")

    print(f"\nTest predictions ({results['n_test']} samples):")
    print(f"  {'Model':<30} {'PPL Act':>7} {'PPL Pred':>8} {'Time Act':>8} {'Time Pred':>9}")
    print("  " + "-" * 70)
    # Combine PPL and time predictions
    ppl_preds = {name: (actual, pred) for name, actual, pred in results['ppl_predictions']}
    for name, time_actual, time_pred in results['time_predictions']:
        ppl_actual, ppl_pred = ppl_preds.get(name, (0, 0))
        print(f"  {name:<30} {ppl_actual:>7.1f} {ppl_pred:>8.1f} {time_actual:>8.1f}s {time_pred:>8.1f}s")

    # Save model
    model_path = Path(__file__).parent.parent / "surrogate_model.pkl"
    model.fit(X, y_ppl, y_time)  # Retrain on all data
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Feature importance for PPL
    print("\nPPL Feature importance (|coefficient|):")
    coefs_ppl = model.weights_ppl[1:]  # Skip bias
    feature_names = ArchFeatures.feature_names()
    importance_ppl = sorted(zip(feature_names, np.abs(coefs_ppl)), key=lambda x: -x[1])
    for name, imp in importance_ppl:
        print(f"  {name:<20} {imp:.3f}")

    # Feature importance for Time
    print("\nTime Feature importance (|coefficient|):")
    coefs_time = model.weights_time[1:]  # Skip bias
    importance_time = sorted(zip(feature_names, np.abs(coefs_time)), key=lambda x: -x[1])
    for name, imp in importance_time:
        print(f"  {name:<20} {imp:.3f}")

    # Example: predict on new candidates
    print("\n" + "=" * 60)
    print("EXAMPLE: Screen hypothetical architectures")
    print("=" * 60)

    candidates = [
        ArchFeatures(n_layers=20, n_kv_heads=2, has_mamba=False, has_linear_attn=False, is_hybrid=False, is_fast_mamba=False),
        ArchFeatures(n_layers=24, n_kv_heads=1, has_mamba=False, has_linear_attn=False, is_hybrid=False, is_fast_mamba=False),
        ArchFeatures(n_layers=16, n_kv_heads=4, has_mamba=False, has_linear_attn=False, is_hybrid=False, is_fast_mamba=False),
        ArchFeatures(n_layers=12, n_kv_heads=8, has_mamba=False, has_linear_attn=False, is_hybrid=False, is_fast_mamba=False),
        ArchFeatures(n_layers=16, n_kv_heads=0, has_mamba=True, has_linear_attn=False, is_hybrid=False, is_fast_mamba=True),
    ]

    X_cand = np.array([c.to_vector() for c in candidates])
    ppl_preds, time_preds = model.predict_both(X_cand)

    print(f"\n{'Architecture':<35} {'Pred PPL':>10} {'Pred Time':>12}")
    print("-" * 60)
    # Sort by PPL
    ranked = sorted(zip(candidates, ppl_preds, time_preds), key=lambda x: x[1])
    for arch, ppl, time in ranked:
        desc = f"L={arch.n_layers}, KV={arch.n_kv_heads}"
        if arch.has_mamba:
            desc += ", Mamba"
        print(f"{desc:<35} {ppl:>10.1f} {time:>10.1f}s")


if __name__ == "__main__":
    main()
