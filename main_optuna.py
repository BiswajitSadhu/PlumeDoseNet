import os
import time
import optuna
import torch
import joblib
import numpy as np
import pandas as pd
import config
from src.utils import setup_windows_fix, log, ensure_directories
from src.data_manager import DataManager
from src.models import TabNetWrapper
from src.evaluation import calculate_metrics, print_metrics, save_metrics
from src.visualization import Visualizer

def objective(trial):
    dm = DataManager(load_artifacts=False)
    df, X, y_norm = dm.load_and_preprocess(config.DATA_PATH, is_training=True)
    X_train, X_val, _, y_train, y_val, _ = dm.get_train_test_val_split(df, X, y_norm)

    # 2. Define Search Space
    n_da = trial.suggest_int("n_d_a", 8, 32, step=8)    ## n_d=n_a always gave better result
    lr = trial.suggest_float("learning_rate", 1e-3, 2e-2, log=True)
    params = {
        'n_d': n_da,
        'n_a': n_da,
        'n_steps': trial.suggest_int("n_steps", 5, 10),
        'gamma': trial.suggest_float("gamma", 1.0, 1.4, log=True),
        'lambda_sparse': trial.suggest_float("lambda_sparse", 1e-6, 1e-5, log=True),
        'optimizer_params': {'lr': lr},
        'mask_type': trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
    }

    # 3. Train Model (Temporary wrapper for trials)
    model_wrapper = TabNetWrapper(
        cat_idxs=dm.cat_idxs, 
        cat_dims=dm.cat_dims, 
        params=params
    )
    
    model_wrapper.train(X_train, y_train, X_val, y_val)
    
    # 4. Return Best Validation RMSE
    best_val_rmse = min(model_wrapper.model.history['val_rmse'])
    return best_val_rmse

def main():
    setup_windows_fix()
    
    # Create specific optuna folder
    optuna_dir = os.path.join(config.OUTPUT_DIR, "optuna")
    ensure_directories([config.OUTPUT_DIR, config.PLOTS_DIR, optuna_dir])
    
    viz = Visualizer(output_dir=f"{config.PLOTS_DIR}/optuna")
    
    # --- 1. Run Optuna Study ---
    db_path = f"sqlite:///{os.path.join(optuna_dir, 'optuna_tab.db')}"
    log(f"🚀 Starting Optuna hyperparameter tuning... (Storage: {db_path})")
    
    study = optuna.create_study(
        direction="minimize", 
        study_name="optuna-tabnet", 
        storage=db_path, 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50) # Set your desired trials

    log("Tuning finished!")
    log(f"Best RMSE: {study.best_value}")
    log(f"Best Params: {study.best_params}")

    # --- 2. Train Final Model with Best Params ---
    log("\n--- Training Final Model with Best Parameters ---")
    
    # Reconstruct parameter dict from best_params
    best_params = study.best_params.copy()
    
    # Handle the special tied parameter logic for n_d/n_a
    if 'n_d_a' in best_params:
        val = best_params.pop('n_d_a')
        best_params['n_d'] = val
        best_params['n_a'] = val
        
    # Handle LR nesting
    if 'learning_rate' in best_params:
        lr = best_params.pop('learning_rate')
        best_params['optimizer_params'] = {'lr': lr}

    # Load Data again for final training
    dm = DataManager(load_artifacts=False)
    df, X, y_norm = dm.load_and_preprocess(config.DATA_PATH, is_training=True)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.get_train_test_val_split(df, X, y_norm)

    # Initialize and Train
    final_model = TabNetWrapper(cat_idxs=dm.cat_idxs, cat_dims=dm.cat_dims, params=best_params)
    start_time = time.time()
    final_model.train(X_train, y_train, X_val, y_val)
    log(f"Final training completed in {time.time() - start_time:.2f} seconds")

    # --- 3. Save and Evaluate ---
    final_model.save(config.TABNET_MODEL_PATH)
    log(f"✅ Final optimized TabNet model saved to {config.TABNET_MODEL_PATH}")

    # Generate Plots (Learning Curve & Masks)
    viz.plot_learning_curves(final_model.model.history, "TabNet_Optimized")
    try:
        feature_names = config.CAT_FEATURES + config.NUM_FEATURES
        viz.plot_tabnet_masks(final_model.model, X_test, feature_names)
    except Exception as e:
        log(f"Could not plot feature masks: {e}")

    # Evaluate on Test Set
    y_pred_norm = final_model.predict(X_test)
    y_true_orig = dm.inverse_transform_target(y_test)
    y_pred_orig = dm.inverse_transform_target(y_pred_norm)

    metrics = calculate_metrics(y_true_orig, y_pred_orig, "TabNet_Optimized")
    print_metrics(metrics)

    # Save Results text
    results_path = os.path.join(config.SAVE_RESULTS_DIR, f"optuna_results_{time.strftime('%Y_%m%d_%H_%M_%S')}.txt")
    save_metrics([metrics], results_path)
    log(f"Final metrics saved to {results_path}")

if __name__ == "__main__":
    main()