"""
Model Persistence Module for Fuhrlander FL2500 Predictive Maintenance
=====================================================================
Save and load all trained ML models for inference in an agentic setup.

Saves to: RUL/saved_models/
  - fuhrlander_fl2500_pm_models.joblib  (all model artifacts)
  - test_data.npz                        (test arrays)
  - test_df.parquet                      (full test DataFrame)
  - metadata.json                        (human-readable manifest)

Usage:
    # After training:
    from save_models import save_all_models
    save_all_models(autoencoder, gmm, binary_clf, multi_clf, ...)

    # For inference:
    from save_models import load_all_models, load_test_data
    models = load_all_models()
    test = load_test_data()
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


def save_all_models(
    autoencoder,
    gmm,
    binary_clf,
    multi_clf,
    lstm_models,
    failure_threshold,
    state_order_map,
    feature_names,
    X_test,
    y_test,
    test_df,
    output_dir=None,
):
    """
    Save all trained model artifacts to disk.

    Parameters
    ----------
    autoencoder : SimpleAutoencoder
        Trained autoencoder with fitted scaler and weights.
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM with 3 components.
    binary_clf : sklearn.ensemble.GradientBoostingClassifier
        Trained binary fault predictor.
    multi_clf : sklearn.ensemble.RandomForestClassifier
        Trained 3-class fault predictor.
    lstm_models : dict
        {state_int: SimpleLSTM} per GMM state.
    failure_threshold : float
        90th percentile of training HI.
    state_order_map : dict
        Maps raw GMM state index to ordered index (0=healthy).
    feature_names : list
        27 feature names (RAW + ENGINEERED).
    X_test : np.ndarray
        Test feature matrix (N, 27).
    y_test : np.ndarray
        Test labels (N,).
    test_df : pd.DataFrame
        Full test DataFrame with timestamps, turbine_id, features, labels.
    output_dir : str, optional
        Save directory. Defaults to RUL/saved_models/.

    Returns
    -------
    dict
        Paths to all saved files.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(output_dir, exist_ok=True)

    # --- Serialize SimpleAutoencoder as dict of numpy arrays ---
    ae_state = {
        'input_dim': autoencoder.input_dim,
        'encoding_dim': autoencoder.encoding_dim,
        'learning_rate': autoencoder.learning_rate,
        'w1': autoencoder.w1, 'b1': autoencoder.b1,
        'w2': autoencoder.w2, 'b2': autoencoder.b2,
        'w3': autoencoder.w3, 'b3': autoencoder.b3,
        'w4': autoencoder.w4, 'b4': autoencoder.b4,
        'w5': autoencoder.w5, 'b5': autoencoder.b5,
        'w6': autoencoder.w6, 'b6': autoencoder.b6,
        'scaler': autoencoder.scaler,
    }

    # --- Serialize SimpleLSTM models as dicts ---
    lstm_state = {}
    for state_key, lstm in lstm_models.items():
        lstm_state[int(state_key)] = {
            'seq_length': lstm.seq_length,
            'training_data': lstm.training_data,
            'mean': float(lstm.mean),
            'std': float(lstm.std),
        }

    # --- Import feature lists for metadata ---
    from wind_turbine_pm_fuhrlander import RAW_FEATURES, ENGINEERED_FEATURES
    
    # --- Get package versions for reproducibility ---
    import sklearn
    import numpy
    import pandas

    # --- Bundle everything into one joblib file ---
    artifacts = {
        'sklearn_version': sklearn.__version__,
        'numpy_version': numpy.__version__,
        'pandas_version': pandas.__version__,
        'autoencoder_state': ae_state,
        'gmm': gmm,
        'binary_clf': binary_clf,
        'multi_clf': multi_clf,
        'lstm_state': lstm_state,
        'failure_threshold': float(failure_threshold),
        'state_order_map': {int(k): int(v) for k, v in state_order_map.items()},
        'feature_names': list(feature_names),
        'raw_features': list(RAW_FEATURES),
        'engineered_features': list(ENGINEERED_FEATURES),
        'feature_engineering_params': {
            'window_size': 24,
            'aggregation_seconds': 3600,
            'pre_fault_hours': 48,
        },
        'model_info': {
            'turbine_model': 'Fuhrlander FL2500 (2.5MW)',
            'train_turbines': [80, 81, 82],
            'test_turbines': [83, 84],
            'n_features': len(feature_names),
            'ae_architecture': f'{len(feature_names)}->64->32->8->32->64->{len(feature_names)}',
            'gmm_n_components': 3,
            'binary_clf_type': type(binary_clf).__name__,
            'multi_clf_type': type(multi_clf).__name__,
            'n_test_samples': int(X_test.shape[0]),
        },
        'saved_at': datetime.now().isoformat(),
    }

    model_path = os.path.join(output_dir, 'fuhrlander_fl2500_pm_models.joblib')
    joblib.dump(artifacts, model_path, compress=3)
    print(f"  Saved model artifacts: {model_path}")

    # --- Save test data (numpy arrays) ---
    test_data_path = os.path.join(output_dir, 'test_data.npz')
    np.savez_compressed(test_data_path, X_test=X_test, y_test=y_test)
    print(f"  Saved test data arrays: {test_data_path}")

    # --- Save test DataFrame (preserves datetime, turbine_id, etc.) ---
    test_df_path = os.path.join(output_dir, 'test_df.parquet')
    test_df.to_parquet(test_df_path, index=False)
    print(f"  Saved test DataFrame: {test_df_path}")

    # --- Save human-readable metadata ---
    metadata = {
        'description': 'Fuhrlander FL2500 Predictive Maintenance - Trained Models',
        'saved_at': datetime.now().isoformat(),
        'files': {
            'fuhrlander_fl2500_pm_models.joblib': 'All model artifacts (AE weights, GMM, classifiers, LSTM, thresholds)',
            'test_data.npz': f'Test feature matrix X_test ({X_test.shape}) and labels y_test ({y_test.shape})',
            'test_df.parquet': f'Full test DataFrame ({len(test_df)} rows) with timestamps, turbine IDs, features, labels',
            'metadata.json': 'This file - human-readable manifest',
        },
        'models': {
            'autoencoder': f'SimpleAutoencoder (numpy, {len(feature_names)}->64->32->8->32->64->{len(feature_names)})',
            'gmm': 'GaussianMixture (sklearn, 3 components)',
            'binary_clf': f'{type(binary_clf).__name__} (sklearn, binary: Healthy vs Anomalous)',
            'multi_clf': f'{type(multi_clf).__name__} (sklearn, 3-class: Healthy/Pre-Fault/Fault)',
            'lstm_models': f'SimpleLSTM (numpy, {len(lstm_models)} state-specific models, seq_length=24)',
        },
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'failure_threshold': float(failure_threshold),
        'train_turbines': [80, 81, 82],
        'test_turbines': [83, 84],
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    return {
        'model_path': model_path,
        'test_data_path': test_data_path,
        'test_df_path': test_df_path,
        'metadata_path': metadata_path,
        'output_dir': output_dir,
    }


def load_all_models(model_dir=None):
    """
    Load all saved model artifacts from disk and reconstruct custom classes.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing saved models. Defaults to RUL/saved_models/.

    Returns
    -------
    dict
        Keys: autoencoder, gmm, binary_clf, multi_clf, lstm_models,
              failure_threshold, state_order_map, feature_names,
              raw_features, engineered_features, feature_engineering_params,
              model_info
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    model_path = os.path.join(model_dir, 'fuhrlander_fl2500_pm_models.joblib')
    print(f"Loading models from: {model_path}")
    artifacts = joblib.load(model_path)

    # --- Reconstruct SimpleAutoencoder ---
    from wind_turbine_pm_fuhrlander import SimpleAutoencoder
    ae_state = artifacts['autoencoder_state']
    autoencoder = SimpleAutoencoder(
        input_dim=ae_state['input_dim'],
        encoding_dim=ae_state['encoding_dim'],
        learning_rate=ae_state['learning_rate'],
    )
    for key in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3',
                'w4', 'b4', 'w5', 'b5', 'w6', 'b6']:
        setattr(autoencoder, key, ae_state[key])
    autoencoder.scaler = ae_state['scaler']

    # --- Reconstruct SimpleLSTM models ---
    from wind_turbine_pm_fuhrlander import SimpleLSTM
    lstm_models = {}
    for state_key, state_data in artifacts['lstm_state'].items():
        lstm = SimpleLSTM(seq_length=state_data['seq_length'])
        lstm.training_data = state_data['training_data']
        lstm.mean = state_data['mean']
        lstm.std = state_data['std']
        lstm_models[int(state_key)] = lstm

    print(f"  Loaded: Autoencoder, GMM, {artifacts['model_info']['binary_clf_type']}, "
          f"{artifacts['model_info']['multi_clf_type']}, {len(lstm_models)} LSTM models")
    print(f"  Failure threshold: {artifacts['failure_threshold']:.6f}")
    print(f"  Features: {artifacts['model_info']['n_features']}")

    return {
        'autoencoder': autoencoder,
        'gmm': artifacts['gmm'],
        'binary_clf': artifacts['binary_clf'],
        'multi_clf': artifacts['multi_clf'],
        'lstm_models': lstm_models,
        'failure_threshold': artifacts['failure_threshold'],
        'state_order_map': artifacts['state_order_map'],
        'feature_names': artifacts['feature_names'],
        'raw_features': artifacts['raw_features'],
        'engineered_features': artifacts['engineered_features'],
        'feature_engineering_params': artifacts['feature_engineering_params'],
        'model_info': artifacts['model_info'],
    }


def load_test_data(model_dir=None):
    """
    Load saved test data for inference demos.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing saved data. Defaults to RUL/saved_models/.

    Returns
    -------
    dict
        Keys: X_test (N,27), y_test (N,), test_df (DataFrame)
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    # Load numpy arrays
    npz_path = os.path.join(model_dir, 'test_data.npz')
    data = np.load(npz_path)
    X_test = data['X_test']
    y_test = data['y_test']

    # Load DataFrame
    df_path = os.path.join(model_dir, 'test_df.parquet')
    test_df = pd.read_parquet(df_path)

    print(f"  Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}, "
          f"test_df ({len(test_df)} rows)")

    return {
        'X_test': X_test,
        'y_test': y_test,
        'test_df': test_df,
    }


if __name__ == '__main__':
    # Quick verification: load models and run a small inference test
    print("=" * 70)
    print("VERIFICATION: Loading saved models and running test inference")
    print("=" * 70)

    models = load_all_models()
    test = load_test_data()

    X_sample = test['X_test'][:5]
    print(f"\nTest input shape: {X_sample.shape}")

    # Autoencoder -> Health Indicator
    hi = models['autoencoder'].get_health_indicator(X_sample)
    print(f"Health Indicators: {hi}")

    # GMM -> States
    raw_states = models['gmm'].predict(hi.reshape(-1, 1))
    states = np.array([models['state_order_map'][s] for s in raw_states])
    print(f"GMM States: {states}")

    # Binary classifier
    binary_pred = models['binary_clf'].predict(X_sample)
    binary_prob = models['binary_clf'].predict_proba(X_sample)
    print(f"Binary predictions: {binary_pred}")
    print(f"Binary P(anomalous): {binary_prob[:, 1]}")

    # Multi-class classifier
    multi_pred = models['multi_clf'].predict(X_sample)
    multi_prob = models['multi_clf'].predict_proba(X_sample)
    print(f"Multi-class predictions: {multi_pred}")
    print(f"Multi-class probabilities:\n{multi_prob}")

    print(f"\nFailure threshold: {models['failure_threshold']:.6f}")
    print(f"Feature names ({len(models['feature_names'])}): {models['feature_names'][:5]}...")
    print("\nVerification PASSED - all models loaded and running correctly.")
