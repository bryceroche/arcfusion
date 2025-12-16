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
    is_fast_mamba = 'mambafast' in name or 'fast' in name

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


def load_training_data(db: ArcFusionDB) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features and targets from training_runs table."""
    runs = db.conn.execute('''
        SELECT model_name, n_layers, d_model, perplexity
        FROM training_runs
        WHERE success = 1 AND perplexity IS NOT NULL
        ORDER BY created_at
    ''').fetchall()

    X = []
    y = []
    names = []

    for model_name, n_layers_db, d_model, ppl in runs:
        features = extract_features(model_name, n_layers_db or 4, d_model or 256)
        X.append(features.to_vector())
        y.append(ppl)
        names.append(model_name)

    return np.array(X), np.array(y), names


class SurrogateModel:
    """Simple surrogate model for PPL prediction."""

    def __init__(self):
        self.weights = None
        self.bias = None
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear regression with standardization."""
        # Standardize features
        self.mean_x = X.mean(axis=0)
        self.std_x = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_x) / self.std_x

        # Standardize target
        self.mean_y = y.mean()
        self.std_y = y.std() + 1e-8
        y_norm = (y - self.mean_y) / self.std_y

        # Add bias term
        X_bias = np.column_stack([np.ones(len(X)), X_norm])

        # Closed-form solution with L2 regularization
        lambda_reg = 0.1
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0  # Don't regularize bias
        self.weights = np.linalg.solve(
            X_bias.T @ X_bias + lambda_reg * I,
            X_bias.T @ y_norm
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict PPL from features."""
        X_norm = (X - self.mean_x) / self.std_x
        X_bias = np.column_stack([np.ones(len(X)), X_norm])
        y_norm = X_bias @ self.weights
        return y_norm * self.std_y + self.mean_y

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'mean_x': self.mean_x,
                'std_x': self.std_x,
                'mean_y': self.mean_y,
                'std_y': self.std_y,
            }, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
            self.mean_x = data['mean_x']
            self.std_x = data['std_x']
            self.mean_y = data['mean_y']
            self.std_y = data['std_y']


def evaluate_model(model: SurrogateModel, X: np.ndarray, y: np.ndarray,
                   names: list[str], split_ratio: float = 0.8):
    """Evaluate model with train/test split."""
    n = len(X)
    n_train = int(n * split_ratio)

    # Use chronological split (older data for training)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    names_test = names[n_train:]

    model.fit(X_train, y_train)

    # Training metrics
    y_train_pred = model.predict(X_train)
    train_mae = np.abs(y_train - y_train_pred).mean()
    train_rmse = np.sqrt(((y_train - y_train_pred) ** 2).mean())

    # Test metrics
    y_test_pred = model.predict(X_test)
    test_mae = np.abs(y_test - y_test_pred).mean()
    test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())

    # Correlation
    corr = np.corrcoef(y_test, y_test_pred)[0, 1] if len(y_test) > 1 else 0

    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'correlation': corr,
        'n_train': n_train,
        'n_test': n - n_train,
        'predictions': list(zip(names_test, y_test, y_test_pred)),
    }


def rank_candidates(model: SurrogateModel, candidates: list[ArchFeatures]) -> list[tuple[ArchFeatures, float]]:
    """Rank candidate architectures by predicted PPL."""
    X = np.array([c.to_vector() for c in candidates])
    preds = model.predict(X)
    ranked = sorted(zip(candidates, preds), key=lambda x: x[1])
    return ranked


def main():
    print("=" * 60)
    print("SURROGATE MODEL FOR PPL PREDICTION")
    print("=" * 60)

    db_path = Path(__file__).parent.parent / "arcfusion.db"
    db = ArcFusionDB(str(db_path))

    # Load data
    print("\nLoading training data...")
    X, y, names = load_training_data(db)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {ArchFeatures.feature_names()}")
    print(f"  PPL range: {y.min():.1f} - {y.max():.1f}")

    # Train and evaluate
    print("\nTraining surrogate model...")
    model = SurrogateModel()
    results = evaluate_model(model, X, y, names, split_ratio=0.8)

    print(f"\nResults:")
    print(f"  Train MAE: {results['train_mae']:.1f} PPL")
    print(f"  Train RMSE: {results['train_rmse']:.1f} PPL")
    print(f"  Test MAE: {results['test_mae']:.1f} PPL")
    print(f"  Test RMSE: {results['test_rmse']:.1f} PPL")
    print(f"  Correlation: {results['correlation']:.3f}")

    print(f"\nTest predictions ({results['n_test']} samples):")
    print(f"  {'Model':<35} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    print("  " + "-" * 60)
    for name, actual, pred in results['predictions']:
        error = pred - actual
        print(f"  {name:<35} {actual:>8.1f} {pred:>8.1f} {error:>+8.1f}")

    # Save model
    model_path = Path(__file__).parent.parent / "surrogate_model.pkl"
    model.fit(X, y)  # Retrain on all data
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Feature importance (coefficient magnitudes)
    print("\nFeature importance (|coefficient|):")
    coefs = model.weights[1:]  # Skip bias
    feature_names = ArchFeatures.feature_names()
    importance = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: -x[1])
    for name, imp in importance:
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

    ranked = rank_candidates(model, candidates)
    print(f"\n{'Architecture':<40} {'Pred PPL':>10}")
    print("-" * 52)
    for arch, ppl in ranked:
        desc = f"L={arch.n_layers}, KV={arch.n_kv_heads}"
        if arch.has_mamba:
            desc += ", Mamba"
        print(f"{desc:<40} {ppl:>10.1f}")


if __name__ == "__main__":
    main()
