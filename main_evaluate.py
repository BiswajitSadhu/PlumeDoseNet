import os
import time
import pandas as pd
import config
from src.utils import setup_windows_fix, log, ensure_directories
from src.data_manager import DataManager
from src.models import XGBoostWrapper, RFWrapper, TabNetWrapper
from src.evaluation import calculate_metrics, print_metrics, save_metrics
from src.visualization import Visualizer

# Define which test file you want to use for evaluation here
TEST_FILE = "Library/filtered_distance_2000_height_200_test_1.csv"

def main():
    metrics_folder = os.path.join(config.SAVE_RESULTS_DIR, "during_testing")
    setup_windows_fix()
    ensure_directories([config.OUTPUT_DIR, config.PLOTS_DIR, metrics_folder])
    viz = Visualizer(output_dir=config.PLOTS_DIR)
    
    # 1. is_training=False ensures we use the SAVED scalers/encoders
    dm = DataManager(load_artifacts=True)
    df_test, X_test, y_test_norm = dm.load_and_preprocess(TEST_FILE, is_training=False)
    
    # We need the original target for error calc
    y_true_orig = df_test[config.TARGET_COL].values
    feature_names = config.CAT_FEATURES + config.NUM_FEATURES

    # 2. Define Models to Evaluate
    models = {
        "XGBoost": (XGBoostWrapper(), config.XGB_MODEL_PATH),
        "RandomForest": (RFWrapper(), config.RF_MODEL_PATH),
        "TabNet": (TabNetWrapper(cat_idxs=dm.cat_idxs, cat_dims=dm.cat_dims), config.TABNET_MODEL_PATH)
    }

    # 3. Run Predictions & Metrics
    results_list = []
    
    for name, (wrapper, path) in models.items():
        if not os.path.exists(path):
            log(f"Skipping {name} - model file not found at {path}")
            continue
            
        log(f"Evaluating {name}...")
        wrapper.load(path)
        
        # Predict (Normalized)
        y_pred_norm = wrapper.predict(X_test)
        
        # Invert to Original Scale
        y_pred_orig = dm.inverse_transform_target(y_pred_norm)
        
        # Store in DataFrame for plotting
        df_test[f'Dose_Pred_{name}'] = y_pred_orig
        df_test[f'Error_{name}'] = df_test[config.TARGET_COL] - y_pred_orig
        
        # Metrics
        met = calculate_metrics(y_true_orig, y_pred_orig, name)
        results_list.append(met)
        print_metrics(met)
        
        # Feature Importance Plot
        try:
            imps = wrapper.get_feature_importance(feature_names)
            viz.plot_feature_importance(imps, name)
        except Exception as e:
            log(f"Could not plot feature importance for {name}: {e}")

        # Random Forest Performance Curve
        if name == "RandomForest":            
            try:
                viz.plot_rf_performance(wrapper.model, X_test, y_test_norm, dm.scaler)
            except Exception as e:
                log(f"Skipping RF performance plot: {e}")

        # Tabnet Feature Masks
        if name == "TabNet":
            try:
                viz.plot_tabnet_masks(wrapper.model, X_test, feature_names) # Add 'num_samples' parameter to modify no. samples masks
            except Exception as e:
                log(f"Skipping TabNet masks: {e}")

    # 4. Generate Comparative Plots
    active_models = [r['Model'] for r in results_list]
    if active_models:
        viz.plot_actual_vs_predicted(df_test, active_models)
        viz.plot_residuals(df_test, active_models)
    models_dict = {name: f'Dose_Pred_{name}' for name in active_models}
    # Plot Violin comparisons for each categorical feature
    for cat_col in config.CAT_FEATURES:
        log(f"Generating violin comparison for {cat_col}...")
        viz.plot_model_comparison_violins(df_test, config.TARGET_COL, models_dict, cat_col)
        
        log(f"Generating MAPE Bar Chart for {cat_col}...")
        viz.plot_error_by_category(df_test, cat_col, active_models)
        
    # 5. Save Text Report
    metrics_file = os.path.join(metrics_folder, f"eval_{time.strftime('%Y_%m%d_%H_%M_%S')}.txt")
    save_metrics(results_list, metrics_file)
    log(f"Evaluation complete. Report saved to {metrics_file}")

if __name__ == "__main__":
    main()